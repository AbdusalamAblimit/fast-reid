# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
import time
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from fastreid.utils import comm
from fastreid.utils.compute_dist import build_dist
from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank_cylib import compile_helper

logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self._cpu_device = torch.device('cpu')

        self._predictions = []
        self._compile_dependencies()

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        prediction = {
            'feats': outputs.to(self._cpu_device, torch.float32),
            'pids': inputs['pid'].to(self._cpu_device),
            'camid': inputs['camid'].to(self._cpu_device)

        }
        self._predictions.append(prediction)

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        features = []
        pids = []
        camids = []
        for prediction in predictions:
            features.append(prediction['feats'])
            pids.append(prediction['pids'])
            camids.append(prediction['camid'])

        features = torch.cat(features, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids, dim=0).numpy()
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = pids[:self._num_query]
        query_camids = camids[:self._num_query]

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = pids[self._num_query:]
        gallery_camids = camids[self._num_query:]

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA

            if self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        from .rank import evaluate_rank
        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
        self._results['mAP'] = mAP * 100
        self._results['mINP'] = mINP * 100
        self._results["metric"] = (mAP + cmc[0]) / 2 * 100

        if self.cfg.TEST.ROC.ENABLED:
            from .roc import evaluate_roc
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)

    def _compile_dependencies(self):
        # Since we only evaluate results in rank(0), so we just need to compile
        # cython evaluation tool on rank(0)
        if comm.is_main_process():
            try:
                from .rank_cylib.rank_cy import evaluate_cy
            except ImportError:
                start_time = time.time()
                logger.info("> compiling reid evaluation cython tool")

                compile_helper()

                logger.info(
                    ">>> done with reid evaluation cython tool. Compilation time: {:.3f} "
                    "seconds".format(time.time() - start_time))
        comm.synchronize()

from fastreid.utils.comm import all_gather, get_world_size, is_main_process
from .rank import evaluate_rank

class PoseReidEvaluator(DatasetEvaluator):
    """
    针对 PoseBaseline 的 Evaluator：
      - process() 接收 inputs, outputs（{'global': out_g, 'local': out_l}）
      - 分别提取 out_g['features'], out_l['features']
      - 收集全局/local/拼接 三套 predictions
      - evaluate() 时对这三套各自跑一次原始的 rank 评估
    """

    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self.num_query = num_query
        self.output_dir = output_dir
        self._cpu = torch.device('cpu')

        # 三套 buffer
        self._preds = {
            'global': [],
            'local': [],
            'concat': [],
        }

        self._compile_dependencies()

    def reset(self):
        for k in self._preds:
            self._preds[k].clear()

    def process(self, inputs, outputs):
        """
        inputs:  dict，包含 'pid' (LongTensor[B]), 'camid' (LongTensor[B])
        outputs: dict，PoseBaseline 在 eval 模式下的返回：
                 {'global': out_g, 'local': out_l}
                 out_* 都是 head 的输出 dict，包含 'features'
        """
        pids = inputs['pid'].to(self._cpu)
        camid = inputs['camid'].to(self._cpu)

        # 抽取特征
        out_g = outputs['global']
        out_l = outputs['local']
        out_c = outputs['concat']
        # out_g = outputs
        # out_l = out_g
        fg = out_g.to(self._cpu)
        fl = out_l.to(self._cpu)
        fc = out_c.to(self._cpu)

        # 同步收集
        for name, feat in [('global', fg), ('local', fl), ('concat', fc)]:
            self._preds[name].append({
                'feats': feat,
                'pids': pids,
                'camid': camid
            })

    def evaluate(self):
        """
        Gather all preds across GPUs, 然后对 'global','local','concat' 各自评估一次
        """
        num_gpus = get_world_size()
        # Gather
        if num_gpus > 1:
            for name in self._preds:
                self._preds[name] = list(itertools.chain(*all_gather(self._preds[name])))
            if not is_main_process():
                return {}
        results = OrderedDict()

        # 对三种特征分别评估
        for name in ['global', 'local', 'concat']:
            preds = self._preds[name]
            # 拼接成大 Tensor
            feats = torch.cat([x['feats'] for x in preds], dim=0)
            pids  = torch.cat([x['pids'] for x in preds], dim=0).numpy()
            camid = torch.cat([x['camid'] for x in preds], dim=0).numpy()

            # 拆分 query / gallery
            qf = feats[:self.num_query]
            q_p = pids[:self.num_query]
            q_c = camid[:self.num_query]
            gf = feats[self.num_query:]
            g_p = pids[self.num_query:]
            g_c = camid[self.num_query:]

            # AQE
            if self.cfg.TEST.AQE.ENABLED:
                logger.info(f"Test with AQE on {name}")
                qf, gf = aqe(qf, gf,
                             self.cfg.TEST.AQE.QE_TIME,
                             self.cfg.TEST.AQE.QE_K,
                             self.cfg.TEST.AQE.ALPHA)

            # 距离
            

            
            dist = build_dist(qf, gf, metric=self.cfg.TEST.METRIC)

            # Rerank
            if self.cfg.TEST.RERANK.ENABLED:
                logger.info(f"Test with rerank on {name}")
                dist = evaluate_rank  # 如果有单独的 rerank 函数可替换此处
                # 或者直接导入 re_ranking 并使用
            if isinstance(dist, torch.Tensor):
                dist = dist.cpu().numpy()
            # Rank / mAP / mINP
            cmc, all_AP, all_INP = evaluate_rank(dist, q_p, g_p, q_c, g_c)
            mAP  = np.mean(all_AP)*100
            mINP = np.mean(all_INP)*100

            # 写入
            for r in [1]:
                results[f"{name}_Rank-{r}"] = cmc[r-1]*100
            results[f"{name}_mAP"]   = mAP

        return copy.deepcopy(results)

    def _compile_dependencies(self):
        # 与原版保持一致，只在主进程编译 Cython
        if is_main_process():
            try:
                from .rank_cylib.rank_cy import evaluate_cy
            except ImportError:
                start = time.time()
                logger.info("> compiling reid evaluation cython tool")
                compile_helper()
                logger.info(f">>> done compiling in {time.time()-start:.3f}s")
        # 同步 barrier
        import torch.distributed as dist
        if get_world_size()>1:
            dist.barrier()