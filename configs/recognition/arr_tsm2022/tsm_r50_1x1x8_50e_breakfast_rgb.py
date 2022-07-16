from linecache import lazycache


_base_ = [
    '../../_base_/models/breakfast_nonaddlayer_r50.py', '../../_base_/schedules/sgd_tsm_50e_wo_scheduler.py',
    '../../_base_/default_runtime.py'
]
# model settings
model = dict(cls_head=dict(num_classes=11))#26
load_from = '/mmaction2/pretrained_models/tsm_r50_256h_1x1x8_50e_sthv2_rgb_20210816-032aa4da.pth'
# dataset settings
#dataset_type = 'VideoDataset'
#data_root = 'data/household/videos'
#data_root_val = 'data/household/videos'
#ann_file_train = 'data/household/household_train_list_videos.txt'
#ann_file_val = 'data/household/household_val_list_videos.txt'
#ann_file_test = 'data/household/household_test_list_videos.txt'
dataset_type = 'VideoDataset'
data_root = 'data/breakfast/videos'
data_root_val = 'data/breakfast/videos'
ann_file_train = 'data/breakfast/annotations/wo_pseudo/breakfast_train_list_videos.txt'# with_pseudo_largedatanum
ann_file_val = 'data/breakfast/annotations/wo_pseudo/breakfast_val_list_videos.txt'
ann_file_test = 'data/breakfast/annotations/wo_pseudo/breakfast_test_list_videos.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),# the definition of 1x1x8 comes from here
    #dict(type='UniformSampleFrames', clip_len=1, num_clips=8),
    dict(type='DecordDecode'),
    #dict(type='Resize', scale=(-1, 256)),
    # dict(
    #     type='MultiScaleCrop',
    #     input_size=224,
    #     scales=(1, 0.875, 0.75, 0.66),
    #     random_crop=False,
    #     max_wh_scale_gap=1,
    #     num_fixed_crops=13,
    #     lazy=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
    #dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),# the definition of 1x1x8 comes from here
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
    #dict(type='Resize', scale=(-1, 256)),
    #dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),# the definition of 1x1x8 comes from here
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
    #dict(type='Resize', scale=(-1, 256)),
    #dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=6,#6
    workers_per_gpu=4,#4
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=2, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    lr=0.0075,  # this lr is used for 8 gpus
    weight_decay=0.0005)
#optimizer = dict(type='SGD', lr=0.075, momentum=0.9, weight_decay=0.00001) 
#Adam
# type='Adam',  # Type of optimizer, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13 for more details
# lr=0.001,  # Learning rate, see detail usages of the parameters in the documentation of PyTorch
# weight_decay=0.0001)  # Weight decay of Adam
# lr_config = dict(
#     policy='CosineAnnealing',
#     by_epoch=False,
#     min_lr=0,
#     warmup='linear',
#     warmup_by_epoch=True,
#     warmup_iters=1,
#     warmup_ratio=0.1)
total_epochs = 50
# runtime settings
work_dir = './work_dirs/experiment20220628_with_pseudo_largedatanum_onlyheader/'
