#_base_ = [
#    '../../_base_/models/tsm_r50.py', '../../_base_/schedules/sgd_tsm_50e.py',
#    '../../_base_/default_runtime.py'
#]
_base_ = [
    '../../_base_/models/household_nonaddlayer_r50.py', '../../_base_/schedules/sgd_tsm_50e.py',
    '../../_base_/default_runtime.py'
]
# model settings
model = dict(cls_head=dict(num_classes=4))#7#11#174
load_from = '/mmaction2/pretrained_models/tsm_r50_1x1x8_50e_sthv1_rgb_20210203-01dce462.pth'  # model path can be found in model zoo
# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/household/rawframes'
data_root_val = 'data/household/rawframes'
ann_file_train = 'data/household/household_train_list_rawframes.txt'
ann_file_val = 'data/household/household_val_list_rawframes.txt'
#ann_file_test = 'data/household/household_test_list_rawframes.txt'
ann_file_test = 'data/household/household_val_list_rawframes.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    #dict(type='MyDebug', note = 'input'),       # use a custom pipeline
    dict(type='VideoAug', degrees = 0, prob = 0),       # use a custom pipeline
    #dict(type='MyDebug', note = 'coloraug'),       # use a custom pipeline
    dict(type='Resize', scale=(-1, 256)),
    #dict(type='MyDebug', note = 'resize'),       # use a custom pipeline
    dict(type='CenterCrop', crop_size=224),     # add center crop
    #dict(type='MyDebug', note = 'centercrop'),       # use a custom pipeline
    dict(
        type='MultiScaleCrop',
        input_size=224,
        #scales=(1, 0.875, 0.75, 0.66),
        #scales=(1,),
        scales=(1, 9.5, 9.0, 0.875, 0.85),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    #dict(type='MyDebug', note = 'scalecrop'),       # use a custom pipeline
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    #dict(type='MyDebug', note = 'resize224'),       # use a custom pipeline
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='MyDebug', note = 'normalize'),       # use a custom pipeline
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=6,
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
    #lr=0.00375,
    weight_decay=0.0005)

# runtime settings
work_dir = './work_dirs/tmp/'
#work_dir = './work_dirs/experiment20210422/debug/'
#work_dir = './work_dirs/experiment20210428_with_handcrop/tsm_r50_1x1x8_50e_household_rgb_usepretrain_nonaddlayer_ignorelaterality/'
#work_dir = './work_dirs/experiment20210421/tsm_r50_1x1x8_50e_household_rgb_usepretrain_nonaddlayer_ignorelaterality/'
#work_dir = './work_dirs/experiment20210420/tsm_r50_1x1x8_50e_household_rgb_usepretrain_nonaddlayer_ignorelaterality/'