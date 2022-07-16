# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTSM',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False, #norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze running stats (mean and var). Default: False.
        shift_div=8),
    cls_head=dict(
        type='HOUSEHOLDHead_NONADDLAYER',
        num_classes=10,#400,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
