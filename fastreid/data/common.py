# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset

from .data_utils import read_image
import pickle
import numpy as np
import warnings

class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "img": img,
            "pid": pid,
            "camid": camid,
            "img_path": img_path,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)


# # MMDetection & MMPose 接口
# from mmdet.apis import init_detector, inference_detector
# from mmpose.apis import init_model as init_pose_estimator, inference_topdown
# from mmengine.registry import init_default_scope
# from mmpose.evaluation.functional import nms
# import torch
# from torchvision.transforms import functional as F
# # ------------------ 检测器 & 姿态模型常量 ------------------ #
# DET_CFG  = "/media/data/abdusalam/torchreid/third-party/mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py"
# DET_CKPT = ("https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
#             "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth")

# HUGE_CFG = ("/media/data/abdusalam/torchreid/third-party/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/"
#             "td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py")
# HUGE_CKPT = ("https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/"
#              "coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth")
# # -------------------------------------------------------- #



# import cv2
# def _draw_gaussian(hm: np.ndarray, center: tuple, sigma: float):
#     x, y = int(center[0]), int(center[1])
#     H, W = hm.shape
#     tmp = int(3 * sigma + 0.5)
#     ul = [max(0, x - tmp), max(0, y - tmp)]
#     br = [min(W, x + tmp + 1), min(H, y + tmp + 1)]
#     if ul[0] >= br[0] or ul[1] >= br[1]:
#         return
#     size = 2 * tmp + 1
#     g = cv2.getGaussianKernel(size, sigma)
#     g = g @ g.T
#     g_y = slice(ul[1] - (y - tmp), br[1] - (y - tmp))
#     g_x = slice(ul[0] - (x - tmp), br[0] - (x - tmp))
#     patch = hm[ul[1]:br[1], ul[0]:br[0]]
#     np.maximum(patch, g[g_y, g_x], out=patch)


# def _build_heatmap_from_kpt(kpt: np.ndarray,
#                             vis: np.ndarray,
#                             img_wh: tuple,
#                             out_size=(64, 48),
#                             sigma=2.0) -> torch.Tensor:
#     """
#     根据关键点坐标和可见性画 heatmap：
#       • kpt: 17×3 numpy (x, y, score)
#       • vis: 17 numpy 可见性分数
#       • img_wh: (W, H)
#     """
#     W, H = img_wh
#     Hh, Wh = out_size
#     hm = np.zeros((17, Hh, Wh), dtype=np.float32)
#     for i in range(17):
#         if vis[i] < 0.5 or kpt[i, 2] <= 0:
#             continue
#         x, y = kpt[i, 0] / W * Wh, kpt[i, 1] / H * Hh
#         _draw_gaussian(hm[i], (x, y), sigma)
#     return torch.from_numpy(hm)

# class CommonPoseDataset(Dataset):
#     """融合 ViTPose 热图 & 自定义 VisPredict 可见性的 Person ReID Dataset"""

#     def __init__(self,
#                  img_items,
#                  transform=None,
#                  relabel=True,
#                  vis_cfg="/media/data/abdusalam/torchreid/third-party/mmpose/mine/config_vispredict.py",
#                  vis_ckpt="/media/data/abdusalam/torchreid/third-party/mmpose/work_dirs/config_vispredict/best_coco_AP_epoch_210.pth",
#                  device="cuda:0",
#                  sigma=2.0,
#                  vis_thr=0.5,
#                  use_raw_heatmap=False,
#                  top_down=True):
#         """
#         img_items: List of (img_path, pid, camid)
#         transform: torchvision or albumentations pipeline，返回 tensor (C,H,W)，值范围 0-255
#         relabel:   是否将原始 pid/camid 重映射到 0...N-1
#         vis_cfg, vis_ckpt: 自定义可见性模型配置与检查点
#         device:    运行检测与姿态的设备
#         sigma:     ViTPose-huge 输出 heatmap 时使用的高斯半径
#         vis_thr:   可见性阈值，用于判断 keypoint 是否可见
#         """
#         self.img_items = img_items
#         self.transform = transform
#         self.relabel = relabel
#         self.device = device
#         self.sigma = sigma
#         self.vis_thr = vis_thr
#         self.use_raw = use_raw_heatmap
#         self.top_down = top_down

#         # 构建 pid/cam 映射
#         pid_set = {i[1] for i in img_items}
#         cam_set = {i[2] for i in img_items}
#         self.pids = sorted(pid_set)
#         self.cams = sorted(cam_set)
#         if relabel:
#             self.pid_dict = {p: idx for idx, p in enumerate(self.pids)}
#             self.cam_dict = {c: idx for idx, c in enumerate(self.cams)}

#         # ------------------ 初始化检测器 & 姿态模型 ------------------ #
#         # 人体检测器 (RTMDet Nano)
#         self.det = init_detector(DET_CFG, DET_CKPT, device=device)

#         # ViTPose-huge：输出原始热图 (17×64×48)
#         self.huge = init_pose_estimator(
#             HUGE_CFG, HUGE_CKPT, device=device,
#             cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=True))))
        
#         # 自定义 VisPredict：只输出 keypoints 可见性
#         self.vis_model = init_pose_estimator(
#             vis_cfg, vis_ckpt, device=device,
#             cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))
#         # ------------------------------------------------------------ #

#     def __len__(self):
#         return len(self.img_items)

#     def __getitem__(self, index):
#         img_path, pid, camid = self.img_items[index]
#         # (1) 读取 & 增强
#         img = read_image(img_path)          # BGR numpy 或 PIL
#         if self.transform:
#             img = self.transform(img)       # Tensor[C,H,W], 0-255

#         # 重标号
#         if self.relabel:
#             pid   = self.pid_dict[pid]
#             camid = self.cam_dict[camid]

#         # (2) 提取热图 & 可见性
#         heatmap, visibility = self._infer_pose(img)

#         return {
#             "img":        img,            # Tensor[C,H,W]
#             "pid":        pid,
#             "camid":      camid,
#             "heatmap":    heatmap,        # Tensor[17, H_img, W_img]
#             "visibility": visibility,     # Tensor[17]
#             "img_path":   img_path,
#         }

#     @property
#     def num_classes(self):
#         return len(self.pids)

#     @property
#     def num_cameras(self):
#         return len(self.cams)

#     def _infer_pose(self, img_tensor):
#         """
#         在增广后的 img_tensor 上运行：
#          1) RTMDet 检测人
#          2) ViTPose-huge 预测原始 heatmaps
#          3) VisPredict 模型预测 keypoint 可见性
#          4) 选取 keypoint 数量最多的实例
#          5) 将 heatmap 从 64×48 上采样×4 到与 img_tensor 相同空间尺寸
#         返回：
#           heatmap：Tensor[17, H_img, W_img]
#           visibility：Tensor[17]
#         """
#         # (a) Tensor → numpy BGR
#         img_np = img_tensor.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)  # H,W,C RGB
#         img_bgr = img_np[..., ::-1]  # 转为 BGR
#         H, W = img_tensor.shape[1:]
#         # (b) 检测
#         if self.top_down:
#             bboxes = np.array([[0, 0, W, H]], dtype=np.float32)
#         else:
        
#             init_default_scope(self.det.cfg.get("default_scope", "mmdet"))
#             det_res = inference_detector(self.det, img_bgr)
#             inst = det_res.pred_instances.cpu().numpy()
#             # 拼接 bboxes + scores, 过滤类别为 person 且 score>0.3
#             bboxes = np.concatenate([inst.bboxes, inst.scores[:, None]], axis=1)
#             bboxes = bboxes[(inst.labels == 0) & (inst.scores > 0.3)]
#             from mmpose.evaluation.functional import nms as pose_nms
#             bboxes = bboxes[pose_nms(bboxes, 0.3)][:, :4]
#         if len(bboxes) == 0:
#             # 没检测到任何人，返回全零结果
#             H, W = img_tensor.shape[1:]
#             return (
#                 torch.zeros(17, H, W, dtype=torch.float32),
#                 torch.zeros(17, dtype=torch.float32)
#             )

#         # (c) ViTPose-huge 提取 heatmaps
#         huge_out = inference_topdown(self.huge, img_bgr, bboxes)
#         # (d) VisPredict 提取可见性
#         vis_out  = inference_topdown(self.vis_model, img_bgr, bboxes)

#         # (d) choose idx
#         if self.top_down:
#             best = 0
#         else:
#             best, _ = max(
#                 enumerate(vis_out),
#                 key=lambda x: int(((x[1].pred_instances.keypoint_scores[0] > self.vis_thr).sum()))
#             )

#         # (e) extract kpt & vis
#         inst_h = huge_out[best].pred_instances
#         kpt   = inst_h.keypoints[0]           # numpy 17×2
#         scr   = inst_h.keypoint_scores[0]     # numpy 17
#         kpt   = np.hstack([kpt, scr[:, None]])
#         inst_v = vis_out[best].pred_instances
#         if hasattr(inst_v, "keypoints_visible"):
#             vis = inst_v.keypoints_visible[0]
#         else:
#             vis = inst_v.keypoint_scores[0]
#         vis = vis.astype(np.float32)

#         # (f) heatmap
#         if self.use_raw:
#             raise NotImplementedError("raw heatmap 支持待补充")
#         else:
#             # 用关键点 & 可见性画高斯
#             hm = _build_heatmap_from_kpt(kpt, vis, (W, H), sigma=self.sigma)
#             # 放大 4 倍
#             # hm = F.interpolate(hm64.unsqueeze(0), scale_factor=4, mode="bilinear", align_corners=False)[0]
#         return hm, torch.from_numpy(vis).float()




# import os
# import time

# # COCO 17 keypoints 的中文名称
# _KEYPOINT_NAMES = [
#     "鼻子", "左眼", "右眼", "左耳", "右耳",
#     "左肩", "右肩", "左肘", "右肘", "左腕",
#     "右腕", "左臀", "右臀", "左膝", "右膝",
#     "左踝", "右踝",
# ]


# def visualize_and_save(img_tensor: torch.Tensor,
#                        heatmap: torch.Tensor,
#                        visibility: torch.Tensor,
#                        save_dir: str = "vis") -> None:
#     """
#     将增强后的 img_tensor、未放大的 heatmap 和 visibility 可见性可视化。
#     在当前工作目录下创建 save_dir，并保存一张拼接了三部分的 PNG：
#       1. 原图
#       2. heatmap（伪彩）
#       3. 原图叠加 heatmap
#     同时在控制台以中文打印每个关键点的可见性。

#     参数：
#       img_tensor: tensor[C, H, W]，值范围 0–255，RGB 格式
#       heatmap:    tensor[17, h, w]，未放大
#       visibility: tensor[17] 或可转为 numpy 的可见性数组
#       save_dir:   保存目录名称
#     """
#     # 1. 确保保存目录存在
#     os.makedirs(save_dir, exist_ok=True)

#     # 2. 从 img_tensor 构建 BGR numpy 原图
#     #    假设 img_tensor 是 RGB 顺序，0–255
#     img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
#     img_bgr = img_np[..., ::-1]  # 转为 BGR

#     H, W = img_bgr.shape[:2]

#     # 3. heatmap 处理：合并最大通道 → 2D uint8
#     hm = heatmap.detach().cpu().numpy() if isinstance(heatmap, torch.Tensor) else heatmap
#     hm_max = np.max(hm, axis=0)                                   # (h, w)
#     hm_norm = (hm_max / (hm_max.max() + 1e-6) * 255).astype(np.uint8)

#     # 4. 上采样到原图尺寸
#     hm_up = cv2.resize(hm_norm, (W, H), interpolation=cv2.INTER_LINEAR)

#     # 5. 伪彩处理
#     hm_color = cv2.applyColorMap(hm_up, cv2.COLORMAP_JET)        # BGR 伪彩

#     # 6. 叠加热图到原图
#     overlay = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)

#     # 7. 拼接三张：原图 | 热图 | 叠加图
#     combined = np.concatenate([img_bgr, hm_color, overlay], axis=1)

#     # 8. 生成时间戳文件名并保存
#     ts = time.strftime("%Y%m%d_%H%M%S")
#     save_path = os.path.join(save_dir, f"{ts}.png")
#     cv2.imwrite(save_path, combined)

#     # 9. 控制台输出可见性
#     vis = visibility.detach().cpu().numpy() if isinstance(visibility, torch.Tensor) else np.array(visibility)
#     print("关键点可见性：")
#     for i, name in enumerate(_KEYPOINT_NAMES):
#         print(f"  {name}: {vis[i]:.2f}")
#     print(f"\n已保存可视化结果 → {save_path}\n")






# class CommonPoseDataset(Dataset):
#     """Generic Person ReID dataset with pose annotations (keypoints, heatmaps, visibility)."""

#     @staticmethod
#     def _load_pickle(path, default=None):
#         """Load a pickle file or return default if not found."""
#         try:
#             with open(path, 'rb') as f:
#                 return pickle.load(f)
#         except FileNotFoundError:
#             return default

#     @staticmethod
#     def _fallback_visibility(pose, thresh=0.5):
#         """If no visibility file, threshold keypoint confidence scores."""
#         conf = pose[:, 2]
#         return (conf > thresh).astype(np.float32).reshape(17, 1)

#     def __init__(self, img_items, transform=None, relabel=True):
#         """
#         Args:
#             img_items: list of tuples (img_path, pid, camid, kpt_path, heat_path, vis_path)
#             transform: callable to apply on the sample dict
#             relabel: whether to remap pids and camids to 0..N-1
#         """
#         self.data = img_items
#         self.transform = transform
#         self.relabel = relabel
#         self.img_items = img_items
#         # collect unique ids
#         pid_set = set()
#         cam_set = set()
#         for item in img_items:
#             pid_set.add(item[1])
#             cam_set.add(item[2])
#         self.pids = sorted(pid_set)
#         self.cams = sorted(cam_set)

#         if relabel:
#             self.pid_dict = {p: i for i, p in enumerate(self.pids)}
#             self.cam_dict = {c: i for i, c in enumerate(self.cams)}

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         # unpack tuple from Market1501Pose: (img, pid, camid, kpt, heat, vis)
#         img_path, pid, camid, kpt_path, heat_path, vis_path = self.data[index]

#         # load image
#         img = read_image(img_path)
#         # if self.transform is not None: img = self.transform(img)
#         # load pose keypoints (17,3)
#         pose = self._load_pickle(kpt_path)
#         if pose is None:
#             raise FileNotFoundError(f"Missing pose file: {kpt_path}")
#         pose = np.asarray(pose, dtype=np.float32)

#         # load heatmap (17,64,48)
#         heatmap = self._load_pickle(heat_path)
#         if heatmap is None:
#             raise FileNotFoundError(f"Missing heatmap file: {heat_path}")
#         heatmap = np.asarray(heatmap, dtype=np.float32)
#         assert heatmap.shape == (17, 64, 48), f"Unexpected heatmap shape: {heatmap.shape}"

#         # load visibility or fallback
#         visibility = self._load_pickle(vis_path)
#         if visibility is None:
#             warnings.warn(f"Missing vis file: {vis_path}, fallback to confidence threshold")
#             visibility = self._fallback_visibility(pose)
#         else:
#             visibility = np.asarray(visibility, dtype=np.float32).reshape(17, 1)

#         # relabel if needed
#         if self.relabel:
#             pid = self.pid_dict[pid]
#             camid = self.cam_dict[camid]

#         sample = {
#             'img': img,
#             'pid': pid,
#             'camid': camid,
#             'img_path': img_path,
#             'pose': pose,
#             'heatmap': heatmap,
#             'visibility': visibility,
#         }

#         if self.transform is not None:
#             sample = self.transform(sample)
#         return sample

#     @property
#     def num_classes(self):
#         return len(self.pids)

#     @property
#     def num_cameras(self):
#         return len(self.cams)
