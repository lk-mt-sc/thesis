# specific
codec = dict(
    type='SPR',
    input_size=(640, 640),
    heatmap_size=(160, 160),
    sigma=(4, 2),
    minimal_diagonal_length=5.656854249492381,
    generate_keypoint_heatmaps=True,
    decode_max_instances=30)

# model
model = dict(
    type='BottomupPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384),
                multiscale_output=True)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth'
        )),
    neck=dict(type='FeatureMapProcessor', concat=True),
    head=dict(
        type='DEKRHead',
        in_channels=720,
        num_keypoints=17,
        num_heatmap_filters=48,
        heatmap_loss=dict(type='KeypointMSELoss', use_target_weight=True),
        displacement_loss=dict(
            type='SoftWeightSmoothL1Loss',
            use_target_weight=True,
            supervise_empty=False,
            beta=0.1111111111111111,
            loss_weight=0.002),
        decoder=codec,
        rescore_cfg=dict(
            in_channels=74,
            norm_indexes=(5, 6),
            init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmpose/pretrain_models/kpt_rescore_coco-33d58c5c.pth'
            ))),
    test_cfg=dict(
        multiscale_test=False,
        flip_test=True,
        nms_dist_thr=0.05,
        shift_heatmap=True,
        align_corners=False))

# dataset
dataset_type = 'CocoDataset'
data_mode = 'bottomup'
data_root = ''

# pipelines
backend_args = dict(backend='local')
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupRandomAffine', input_size=(640, 640)),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='GenerateTarget',
        encoder=codec),
    dict(type='BottomupGetHeatmapMask'),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize',
        input_size=(640, 640),
        size_factor=32,
        resize_mode='expand'),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'crowd_index', 'ori_shape',
                   'img_shape', 'input_size', 'input_center', 'input_scale',
                   'flip', 'flip_direction', 'flip_indices', 'raw_ann_info',
                   'skeleton_links'))
]
test_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize',
        input_size=(640, 640),
        size_factor=32,
        resize_mode='expand'),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'crowd_index', 'ori_shape',
                   'img_shape', 'input_size', 'input_center', 'input_scale',
                   'flip', 'flip_direction', 'flip_indices', 'raw_ann_info',
                   'skeleton_links'))
]

# dataloaders
train_dataloader = dict(
    batch_size=10,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline))

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/sc/annotations/person_keypoints_val2017.json',
    nms_mode='none',
    score_mode='keypoint')
test_evaluator = dict(
    type='CocoMetric',
    format_only=True,
    ann_file='',
    nms_mode='none',
    score_mode='keypoint',
    outfile_prefix=''
)

# loop configs
train_cfg = dict(by_epoch=True, max_epochs=140, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=140,
        milestones=[90, 120],
        gamma=0.1,
        by_epoch=True)
]

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.001))
auto_scale_lr = dict(base_batch_size=80)

# hooks
default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='coco/AP',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False))
custom_hooks = [dict(type='SyncBuffersHook')]

# runtime configs
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = ''
resume = False
