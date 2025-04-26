# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        self.heads = heads

        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        return {
            'backbone': backbone,
            'heads': heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "pid" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["pid"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)
            return losses
        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['img']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        return loss_dict


from mmengine.config import Config
from mmpose.models import build_posenet
from mmengine.runner import load_checkpoint
from mmpose.utils import register_all_modules as register_all_modules_mmpose
from mmpose.models.data_preprocessors import PoseDataPreprocessor
register_all_modules_mmpose()




POSE_CFG = "pose/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py"
POSE_CKPT = ("https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/"
             "coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth")

VIS_CFG = "pose/config_vispredict.py"
VIS_CKPT = "pretrained/best_coco_AP_epoch_210.pth"


@META_ARCH_REGISTRY.register()
class PoseBaseline(nn.Module):
    """
    this interface is experimental .
    带姿态信息的Baseline模型
      - backbone 接收 (img, heatmap, visibility)，输出全局和局部特征
      - heads 分为 global_head 和 local_head分别对两种特征做分类与度量学习
    """

    @configurable
    def __init__(
        self,
        *,
        backbone,
        global_head,
        local_head,
        pixel_mean,
        pixel_std,
        pose_net,
        device,
        loss_kwargs=None
       ):
        super().__init__()
        # 主干网络，输入 img, heatmap, visibility
        self.backbone = backbone
        # 分类与度量头：全局、局部
        self.global_head = global_head
        self.local_head = local_head
        # 损失超参
        self.loss_kwargs = loss_kwargs
        # 图像归一化参数
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)
        

        self.pose_net = pose_net
        for name, param in pose_net.named_parameters():
            param.requires_grad = False
            
            
            
        self.pose_mean = torch.tensor([123.675, 116.28,  103.53], device=device) \
                    .view(1,3,1,1)
        self.pose_std  = torch.tensor([ 58.395,  57.12,   57.375], device=device) \
                            .view(1,3,1,1)
        self._vis_counter = 0


    @classmethod
    def from_config(cls, cfg):
        # 根据 cfg 构建 backbone 和 heads
        backbone = build_backbone(cfg)
        # 假设 cfg.MODEL.GLOBAL_HEAD / LOCAL_HEAD 配置
        global_head = build_heads(cfg)
        local_head = build_heads(cfg)
        pose_cfg = Config.fromfile(VIS_CFG)
        posenet = build_posenet(pose_cfg.model)
        ckpt = load_checkpoint(posenet, VIS_CKPT, map_location='cpu')
        
        
        return {
            'backbone': backbone,
            'global_head': global_head,
            'local_head': local_head,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs': cfg.MODEL.LOSSES,
            'pose_net': posenet,
            'device':  cfg.MODEL.DEVICE
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        
        
        # images_raw = batched_inputs['img']  # Tensor[B,C,H,W] 0-255 or 0-1
        # ——— 走 PoseNet 应该有的预处理 ———
        # PackPoseInputs 之后，PoseDataPreprocessor 接受一个 dict，返回 dict:
        # images_for_pose = (images_raw - self.pose_mean) / self.pose_std   # N×3×256×192


        # 对输入图像进行归一化：减均值除以标准差
        images = batched_inputs['img']  # Tensor[B,C,H,W]
        images = images.to(self.device)
        images = images.sub_(self.pixel_mean).div_(self.pixel_std)
        
        return images

    def forward(self, batched_inputs):
        """
        前向：训练阶段返回两部分损失字典，测试阶段返回特征和分类结果
        """
        # 准备输入
        # images_raw = batched_inputs["img"].clone()
        
        images= self.preprocess_image(batched_inputs)
        with torch.no_grad():
            pose_res = self.pose_net.backbone(images)
            pose_res = self.pose_net.head(pose_res)
        heatmap = pose_res[0]
        visibility = pose_res[1]

        # self._vis_counter += 1
        # if self._vis_counter %10 == 0 and not self.training:
        #     # images_for_vis: 需要是 0–255 uint8
        #     # 如果你 preprocess_image 里把数据拉到 [0,1]，这里要先 *255
        #     imgs = images_raw.cpu().byte()
        #     visualize_batch_and_save(
        #         imgs,
        #         heatmaps=heatmap.cpu(),
        #         visibility=visibility.cpu(),
        #         save_dir='./vis'
        #     )
        
        
        feat_global, feat_local = self.backbone(images, heatmap, visibility)
        # feat_global = self.backbone(images)
        # feat_local = feat_global

        if self.training:
            pid = batched_inputs['pid'].to(self.device)
            # 全局头输出
            out_g = self.global_head(feat_global, pid)
            # 局部头输出
            out_l = self.local_head(feat_local, pid)
            # 分别计算损失
            loss_g = self._compute_losses(out_g, pid, prefix='global')
            loss_l = self._compute_losses(out_l, pid, prefix='local')
            # 合并两个损失字典
            loss_dict = {}
            loss_dict.update(loss_g)
            loss_dict.update(loss_l)
            return loss_dict
        else:
            # 测试阶段只返回两种输出
            out_g = self.global_head(feat_global)
            out_l = self.local_head(feat_local)
            return {'global': out_g, 'local': out_l}
            # return out_g

    def _compute_losses(self, outputs, targets, prefix=''):
        """
        根据 outputs 计算交叉熵、三元组等损失，prefix 用于区分全局/局部
        outputs 中应包含:
           - 'pred_class_logits'
           - 'cls_outputs'
           - 'features'
        """
        loss_dict = {}
        # 交叉熵
        if 'CrossEntropyLoss' in self.loss_kwargs.NAME:
            ce_params = self.loss_kwargs.CE
            loss_ce = cross_entropy_loss(
                outputs['cls_outputs'], targets,
                ce_params.EPSILON, ce_params.ALPHA
            ) * ce_params.SCALE
            loss_dict[f'{prefix}_loss_cls'] = loss_ce
        # 三元组
        if 'TripletLoss' in self.loss_kwargs.NAME:
            tri_params = self.loss_kwargs.TRI
            loss_tri = triplet_loss(
                outputs['features'], targets,
                tri_params.MARGIN, tri_params.NORM_FEAT, tri_params.HARD_MINING
            ) * tri_params.SCALE
            loss_dict[f'{prefix}_loss_triplet'] = loss_tri
        # CircleLoss
        if 'CircleLoss' in self.loss_kwargs.NAME:
            cir_params = self.loss_kwargs.CIRCLE
            loss_circle = pairwise_circleloss(
                outputs['features'], targets,
                cir_params.MARGIN, cir_params.GAMMA
            ) * cir_params.SCALE
            loss_dict[f'{prefix}_loss_circle'] = loss_circle
        # Cosface
        if 'Cosface' in self.loss_kwargs.NAME:
            cos_params = self.loss_kwargs.COSFACE
            loss_cos = pairwise_cosface(
                outputs['features'], targets,
                cos_params.MARGIN, cos_params.GAMMA
            ) * cos_params.SCALE
            loss_dict[f'{prefix}_loss_cosface'] = loss_cos
        return loss_dict

import os
import time
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
# 英文 COCO 17 keypoint 名称
_KEYPOINT_NAMES_EN = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
def visualize_batch_and_save(
    img_tensors: torch.Tensor,
    heatmaps:    torch.Tensor,
    visibility:  torch.Tensor,
    save_dir:    str = "vis"
) -> None:
    """
    将一个 batch 可视化为 4×B 格式并保存：
      row0 原图(RGB→BGR)
      row1 伪彩热图
      row2 原图+热图
      row3 文本( key: value )
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. 转 numpy uint8
    imgs = img_tensors.detach().cpu().numpy()
    # 如果值在 [0,1]，先 *255
    if imgs.max() <= 1.01:
        imgs = (imgs * 255).astype(np.uint8)
    else:
        imgs = imgs.astype(np.uint8)
    B, C, H, W = imgs.shape

    # 2. heatmaps、visibility
    hms = heatmaps.detach().cpu().numpy()   # [B,17,h,w]
    vis = visibility.detach().cpu().numpy() # [B,17]

    rows = [ [] for _ in range(4) ]  # 四行，每行 put B 张图
    font = ImageFont.load_default()

    for i in range(B):
        # 原图：RGB→BGR
        img_rgb = imgs[i].transpose(1,2,0)      # H,W,3
        img_bgr = img_rgb[..., ::-1].copy()      # H,W,3 BGR
        rows[0].append(img_bgr)

        # 伪彩热图：先取通道 max，再 resize & applyColorMap
        hm = hms[i].max(axis=0)                 # h,w
        hm_norm = (hm / (hm.max()+1e-6) * 255).astype(np.uint8)
        hm_up = cv2.resize(hm_norm, (W, H), interpolation=cv2.INTER_LINEAR)
        hm_color = cv2.applyColorMap(hm_up, cv2.COLORMAP_JET)  # H,W,3 BGR
        rows[1].append(hm_color)

        # 叠加
        overlay = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)
        rows[2].append(overlay)

        # 文本行：PIL 画白底 + 文本 → 转 BGR
        txt_img = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(txt_img)
        lines = [f"{_KEYPOINT_NAMES_EN[k]}:{vis[i,k]:.2f}"
                 for k in range(17)]
        # 每行文本高度约 12px，根据 H 分配行距
        line_h = H // 18
        y = 0
        for line in lines:
            draw.text((5, y), line, fill="black", font=font)
            y += line_h
        # PIL→numpy RGB
        txt_np = np.array(txt_img)
        # 转 BGR
        txt_bgr = txt_np[..., ::-1].copy()
        rows[3].append(txt_bgr)

        # 控制台也可以打印一下
        # print(f"Sample {i} visibility:")
        # for k, name in enumerate(_KEYPOINT_NAMES_EN):
        #     print(f"  {name}: {vis[i,k]:.2f}")
        # print("-"*40)

    # 3. 拼接：先每行水平拼接，再纵向拼接四行
    horz = [ np.concatenate(r, axis=1) for r in rows ]  # 四个大图
    big = np.concatenate(horz, axis=0)                  # (4H) x (BW) x 3

    # 4. 保存
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"{ts}.png")
    cv2.imwrite(path, big)
    # print(f"Saved batch vis → {path}\n")