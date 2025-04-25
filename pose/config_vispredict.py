# config_vispredict.py

_base_ = ['default_runtime.py']

# runtime
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10)
auto_scale_lr = dict(base_batch_size=512)
backend_args = dict(backend='local')

codec = dict(
    type='UDPHeatmap',
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2)

custom_hooks = [
    dict(type='SyncBuffersHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'])

data_mode = 'topdown'
data_root = '/media/data/dataset/coco2017/'
dataset_type = 'CocoDataset'

default_hooks = dict(
    badcase=dict(
        type='BadCaseAnalysisHook',
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=1,
        rule='greater',
        save_best='coco/AP'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook')
)

default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)

# 修改 model，使用 VisPredictHead 包装器（同时保留 HeatmapHead 的热图预测功能）
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='base',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.55,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_base_20230913.pth'),
    ),
    # 注意：在 head 中使用 VisPredictHead 来包装 HeatmapHead，
    # 这样模型将同时预测热图与关键点可见性。损失配置中 BCELoss 用于监督可见性预测分支。
    head=dict(
        type='VisPredictHead',
        loss=dict(
            type='BCELoss',
            use_target_weight=True,
            use_sigmoid=True,
            loss_weight=1e-3),
        pose_cfg=dict(
            type='HeatmapHead',
            in_channels=768,
            out_channels=17,
            deconv_out_channels=(256, 256),
            deconv_kernel_sizes=(4, 4),
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
            decoder=codec)),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    constructor='LayerDecayOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=32,
        layer_decay_rate=0.85,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'norm': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0)
        },
    ),
)

param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# 数据管道配置
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file='/media/data/abdusalam/torchreid/third-party/mmpose/data/coco/person_detection_result/COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=vis_backends)
