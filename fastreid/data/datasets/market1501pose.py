# coding: utf-8
import glob
import os
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Market1501Pose(ImageDataset):
    """Market1501 with 2D-pose annotations (keypoints, heatmaps, visibility).

    要求目录结构：
    ├── bounding_box_train
    ├── query
    ├── bounding_box_test
    └── pose
        ├── bounding_box_train
        ├── query
        ├── bounding_box_test
        └── (可选) images

    每个 pose 子目录下应有与图片同名的：
      • *_kpt.pkl
      • *_heat.pkl
      • *_vis.pkl
    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_name = 'market1501'

    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        self.root = root
        # 找到数据根目录
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        data_dir = osp.join(self.dataset_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                f'Expected folder "Market-1501-v15.09.15" under {self.dataset_dir}, '
                'falling back to top-level.'
            )
            self.data_dir = self.dataset_dir

        # 原始图像子目录
        self.train_dir   = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir   = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        # pose 文件所在目录
        self.pose_root = osp.join(self.data_dir, 'pose')
        self.train_pose_dir   = osp.join(self.pose_root, 'bounding_box_train')
        self.query_pose_dir   = osp.join(self.pose_root, 'query')
        self.gallery_pose_dir = osp.join(self.pose_root, 'bounding_box_test')
        self.extra_gallery_pose_dir = osp.join(self.pose_root, 'images')

        required = [self.data_dir,
                    self.train_dir, self.query_dir, self.gallery_dir,
                    self.pose_root]
        if self.market1501_500k:
            required.append(self.extra_gallery_dir)
        self.check_before_run(required)

        train = lambda: self.process_dir(self.train_dir,   self.train_pose_dir,   is_train=True)
        query = lambda: self.process_dir(self.query_dir,   self.query_pose_dir,   is_train=False)
        gallery = lambda: (
            self.process_dir(self.gallery_dir, self.gallery_pose_dir, is_train=False)
            + (self.process_dir(self.extra_gallery_dir,
                                self.extra_gallery_pose_dir,
                                is_train=False)
               if self.market1501_500k else [])
        )

        super(Market1501Pose, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, img_dir, pose_dir, is_train=True):
        img_paths = glob.glob(osp.join(img_dir, '*.jpg'))
        # 文件名示例：0001_c1_000151.jpg
        pattern = re.compile(r'([-\d]+)_c(\d)')
        data = []
        for img_path in img_paths:
            m = pattern.search(osp.basename(img_path))
            if not m:
                continue
            pid, camid = map(int, m.groups())
            if pid == -1:
                continue  # 忽略 junk 图像
            assert 0 <= pid <= 1501
            assert 1 <= camid <= 6
            camid -= 1  # camid 从 0 开始

            if is_train:
                pid   = f'{self.dataset_name}_{pid}'
                camid = f'{self.dataset_name}_{camid}'

            base = osp.splitext(osp.basename(img_path))[0]
            # 拼出 pose 文件路径
            kpt_path  = osp.join(pose_dir, f'{base}_kpt.pkl')
            heat_path = osp.join(pose_dir, f'{base}_heat.pkl')
            vis_path  = osp.join(pose_dir, f'{base}_vis.pkl')

            data.append((img_path, pid, camid, kpt_path, heat_path, vis_path))
        return data
