import random, math, numbers
import numpy as np, torch
from torchvision.transforms import functional as F
from torchvision import transforms as T
from typing import Union
HM_RATIO = 0.25          # 17-channel heat-map 缩放比

# 根据你给出的 17-点顺序
FLIP_PAIRS = [
    (1, 2),   # eye
    (3, 4),   # ear
    (5, 6),   # shoulder
    (7, 8),   # elbow
    (9,10),   # wrist
    (11,12),  # hip
    (13,14),  # knee
    (15,16),  # ankle
]



def _update_visibility(vis, kpt, w, h):
    """若关键点落到裁剪框外，则将 vis 置 0"""
    x, y = kpt[:, 0], kpt[:, 1]
    vis[(x < 0) | (x >= w) | (y < 0) | (y >= h)] = 0
    return vis




class ResizePose:
    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)
        self.interp = interpolation
    def __call__(self, sample):
        img, hm, kpt, vis = sample["img"], sample["heatmap"], sample["pose"], sample["visibility"]

        w0, h0 = img.size
        img = F.resize(img, self.size, self.interp)
        w1, h1 = img.size
        scale_x, scale_y = w1 / w0, h1 / h0

        kpt = kpt.astype(np.float32)
        kpt[:, 0] *= scale_x
        kpt[:, 1] *= scale_y

        # heatmap 同比例
        if isinstance(hm, np.ndarray):
            hm = torch.from_numpy(hm)
        hm_size = (int(self.size[0] * HM_RATIO), int(self.size[1] * HM_RATIO))  # (Hh, Wh)
        hm = F.resize(hm, hm_size, self.interp)

        sample.update(img=img, heatmap=hm, kpt=kpt, vis=_update_visibility(vis, kpt, w1, h1))
        return sample

class PadPose:
    def __init__(self, padding, fill=0, padding_mode="constant"):
        if isinstance(padding, numbers.Number):
            self.pad = (padding, padding, padding, padding)
        elif len(padding) == 2:
            self.pad = (padding[0], padding[1], padding[0], padding[1])
        else:
            self.pad = tuple(padding)  # l,t,r,b
        self.fill, self.mode = fill, padding_mode
    def __call__(self, sample):
        img, hm, kpt, vis = sample["img"], sample["heatmap"], sample["pose"], sample["visibility"]
        l,t,r,b = self.pad

        img = F.pad(img, self.pad, self.fill, self.mode)
        kpt = kpt.copy(); kpt[:, 0] += l; kpt[:, 1] += t
        hm = F.pad(hm, tuple(int(p*HM_RATIO) for p in self.pad), self.fill, self.mode)

        w, h = img.size
        sample.update(img=img, heatmap=hm, kpt=kpt, vis=_update_visibility(vis, kpt, w, h))
        return sample


class RandomResizedCropPose:
    def __init__(self, size, scale=(0.08,1.0), ratio=(3./4.,4./3.), interpolation=F.InterpolationMode.BILINEAR):
        self.size = size if isinstance(size, (list,tuple)) else (size,size)
        self.scale, self.ratio, self.interp = scale, ratio, interpolation
    def __call__(self, sample):
        img, hm, kpt, vis = sample["img"], sample["heatmap"], sample["pose"], sample["visibility"]
        i, j, h, w = T.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        # i,j,h,w = F.get_params(img, self.scale, self.ratio)  # crop box
        img = F.resized_crop(img, i, j, h, w, self.size, self.interp)

        # heatmap等比例 box
        i_hm  = int(i * HM_RATIO); j_hm = int(j * HM_RATIO)
        h_hm  = int(h * HM_RATIO); w_hm = int(w * HM_RATIO)
        hm    = F.resized_crop(hm, i_hm, j_hm, h_hm, w_hm,
                               (int(self.size[0]*HM_RATIO), int(self.size[1]*HM_RATIO)),
                               self.interp)

        kpt = kpt.copy(); kpt[:, 0] -= j; kpt[:, 1] -= i
        kpt[:, 0] *= self.size[0]/w; kpt[:, 1] *= self.size[1]/h

        vis = _update_visibility(vis, kpt, self.size[0], self.size[1])
        sample.update(img=img, heatmap=hm, kpt=kpt, vis=vis)
        return sample
    

class RandomHorizontalFlipPose:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample

        img, hm, kpt, vis = sample["img"], sample["heatmap"], sample["pose"], sample["visibility"]

        # 1) 图像&热图做空间翻转
        img = F.hflip(img)
        hm  = torch.flip(hm, [-1])   # width 维度

        # 2) 交换左右通道（heatmap、坐标、可见性）
        for i, j in FLIP_PAIRS:
            # heatmap 通道交换
            hm[i], hm[j] = hm[j].clone(), hm[i].clone()
            # 坐标交换
            kpt[[i, j]] = kpt[[j, i]]
            # 可见性交换
            vis[i], vis[j] = vis[j], vis[i]

        # 3) 更新坐标的 x 值
        w, _ = img.size
        kpt[:, 0] = w - 1 - kpt[:, 0]

        sample.update(img=img, heatmap=hm, kpt=kpt, vis=vis)
        return sample


class RandomErasingPose:
    def __init__(self, prob=0.5, value=0.0):
        """
        value:
          • float / int            → 单通道同填
          • tuple/list 长度=C      → 按通道填
          • torch.Tensor shape=(C,)→ 同上
        """
        self.prob  = prob
        self.value = value

    # -------------------------------------------------
    def _value_tensor(self, img):
        """
        返回 (C,1,1) 或 0-dim Tensor，可被广播到 img[:, y1:y2, x1:x2]
        """
        C, _, _ = img.shape
        if isinstance(self.value, torch.Tensor):
            v = self.value.to(img).flatten()
        elif isinstance(self.value, numbers.Number):
            v = torch.tensor([self.value]*C, dtype=img.dtype, device=img.device)
        elif isinstance(self.value, (tuple, list)):
            if len(self.value) != C:
                raise ValueError(f"value length {len(self.value)} "
                                 f"does not match channel {C}")
            v = torch.tensor(self.value, dtype=img.dtype, device=img.device)
        else:
            raise TypeError("Unsupported type for `value`")

        return v.view(C, 1, 1)          # broadcast-friendly

    # -------------------------------------------------
    def __call__(self, sample):
        if torch.rand(1).item() > self.prob:
            return sample

        img, hm, kpt, vis = (
            sample["img"],      # Tensor C×H×W
            sample["heatmap"],  # Tensor 17×Hh×Wh
            sample["kpt"],      # Tensor 17×2
            sample["vis"],      # Tensor 17
        )

        _, H, W = img.shape
        area = H * W

        # ---- 随机区域 ----
        erase_area = torch.empty(1).uniform_(0.02, 0.4).item() * area
        aspect     = torch.empty(1).uniform_(0.3, 3.3).item()
        eh = int(round((erase_area / aspect) ** 0.5))
        ew = int(round((erase_area * aspect) ** 0.5))
        if eh >= H or ew >= W:
            return sample                     # 异常尺寸，跳过

        y1 = torch.randint(0, H - eh + 1, (1,)).item()
        x1 = torch.randint(0, W - ew + 1, (1,)).item()
        y2, x2 = y1 + eh, x1 + ew

        # ---- 图像擦除 ----
        img[:, y1:y2, x1:x2] = self._value_tensor(img)

        # ---- 热图擦除 ----
        # hy1, hx1 = int(y1 * HM_RATIO), int(x1 * HM_RATIO)
        # hy2, hx2 = int(y2 * HM_RATIO), int(x2 * HM_RATIO)
        # hm[:, hy1:hy2, hx1:hx2] = 0.0

        # ---- 可见性更新 ----
        inside = ((kpt[:, 0] >= x1) & (kpt[:, 0] < x2) &
                  (kpt[:, 1] >= y1) & (kpt[:, 1] < y2))
        vis[inside] = 0.0

        sample.update(img=img, heatmap=hm, kpt=kpt, vis=vis)
        return sample
    
class ToTensorPose:
    """
    • 图像  : PIL / ndarray  → FloatTensor (C,H,W)，归一化到 [0,1] 并减均值除方差  
    • 热图  : ndarray / Tensor → FloatTensor (17,Hh,Wh)  
    • 关键点: ndarray           → FloatTensor (17,2)  
    • 可见性: ndarray / Tensor → FloatTensor (17,)
    """
    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std  = torch.tensor(std).view(3, 1, 1)

    def __call__(self, sample):
        img, hm, kpt, vis = (
            sample["img"],
            sample["heatmap"],
            sample["pose"],
            sample["visibility"],
        )

        # -------- 1) Image → tensor & normalize --------
        if isinstance(img, np.ndarray):        # H×W×C, uint8
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div_(255.)
        else:                                  # PIL.Image
            img = F.to_tensor(img)
        img = (img - self.mean) / self.std

        # -------- 2) Heat-map → float tensor ----------
        if isinstance(hm, np.ndarray):
            hm = torch.from_numpy(hm)
        hm = hm.float()

        # -------- 3) Keypoints / Visibility -----------
        kpt = torch.as_tensor(kpt, dtype=torch.float32)
        vis = torch.as_tensor(vis, dtype=torch.float32)

        sample.update(img=img, heatmap=hm, kpt=kpt, vis=vis)
        return sample
    
    
class ComposePose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample