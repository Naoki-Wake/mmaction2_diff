# model settings
model = dict(
    type='TEM',
    temporal_dim=100,
    boundary_ratio=0.1,
    tem_feat_dim=800,
    tem_hidden_dim=512,
    tem_match_threshold=0.5)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='score')
# dataset settings
dataset_type = 'ActivityNetDataset'
data_root = '/dataset/maction_feat/'#'/dataset/activitynet/maction_feat/'
data_root_val = '/dataset/maction_feat/'
ann_file_train = '/dataset/annotation/anet_anno_train.json'
ann_file_val = '/dataset/annotation/anet_anno_test.json'
ann_file_test = '/dataset/annotation/anet_anno_test.json'

work_dir = 'work_dirs/bsn_800x100_20e_1x16_activitynet_feature_train/'
tem_results_dir = f'{work_dir}/tem_results/'

test_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(
        type='Collect',
        keys=['raw_feature'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature'])
]
train_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(type='ToDataContainer', fields=[dict(key='gt_bbox', stack=False)])
]
val_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(type='ToDataContainer', fields=[dict(key='gt_bbox', stack=False)])
]

data = dict(
    videos_per_gpu=1,
    workers_per_gpu=8,
    train_dataloader=dict(drop_last=True),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=data_root_val),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root))

# optimizer
optimizer = dict(
    type='Adam', lr=0.001, weight_decay=0.0001)  # this lr is used for 1 gpus

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=7)

total_epochs = 20
checkpoint_config = dict(interval=1, filename_tmpl='tem_epoch_{}.pth')

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
output_config = dict(out=tem_results_dir, output_format='csv')
