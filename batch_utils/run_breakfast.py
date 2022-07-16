from mmcv import Config, DictAction
from numpy import argsort
from mmaction.apis import inference_recognizer, init_recognizer
import os.path as osp
import os
import time
from glob import glob
import argparse
import numpy as np
import mmaction_diff
import mmaction_diff.models.heads.household_head_nonaddlayer

import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmaction.models.builder import HEADS
from mmaction.models.heads.base import AvgConsensus, BaseHead


@HEADS.register_module()
class HOUSEHOLDHead_NONADDLAYER(BaseHead):
    """Class head for HOUSEHOLD on top of TSM.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        num_segments (int): Number of frame segments. Default: 8.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        is_shift (bool): Indicating whether the feature is shifted.
            Default: True.
        temporal_pool (bool): Indicating whether feature is temporal pooled.
            Default: False.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_segments=8,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.8,
                 init_std=0.001,
                 is_shift=True,
                 temporal_pool=False,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.num_segments = num_segments
        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, num_segs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Useless in TSMHead. By default, `num_segs`
                is equal to `clip_len * num_clips * num_crops`, which is
                automatically generated in Recognizer forward phase and
                useless in TSM models. The `self.num_segments` we need is a
                hyper parameter to build TSM models.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N * num_segs, in_channels, 1, 1]
        x = torch.flatten(x, 1)
        # [N * num_segs, in_channels]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N * num_segs, num_classes]
        cls_score = self.fc_cls(x)
        
        if self.is_shift and self.temporal_pool:
            # [2 * N, num_segs // 2, num_classes]
            cls_score = cls_score.view((-1, self.num_segments // 2) +
                                       cls_score.size()[1:])
        else:
            # [N, num_segs, num_classes]
            cls_score = cls_score.view((-1, self.num_segments) +
                                       cls_score.size()[1:])
        # [N, 1, num_classes]
        cls_score = self.consensus(cls_score)
        # [N, num_classes]
        return cls_score.squeeze(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run breakfast')
    parser.add_argument('--dir-root', default='/tmp/repo', type=str)
    parser.add_argument(
        '--config',
        default='configs/recognition/arr_tsm2022/tsm_r50_1x1x8_50e_breakfast_rgb.py',
        type=str)
    parser.add_argument(
        '--load-from',
        default='/lfovision_sthv2_breakfast/pretrained_models/tsm_r50_256h_1x1x8_50e_sthv2_rgb_20210816-032aa4da.pth',
        type=str)
    parser.add_argument(
        '--work-dir-root',
        default='/lfovision_log/tsm_learningrate/',
        type=str)
    parser.add_argument('--work-dir-name', default='', type=str)
    parser.add_argument(
        '--train-file-dir',
        default='/lfovision_sthv2_breakfast/annotations/with_pseudo_largedatanum/',
        type=str)
    parser.add_argument(
        '--dir-videos-root',
        default='/lfovision_sthv2_breakfast/',
        type=str)
    parser.add_argument('--videos-per-gpu', default=80, type=int)
    parser.add_argument('--workers-per-gpu', default=20, type=int)
    parser.add_argument('--lr', default=0.0075, type=float)
    parser.add_argument('--weight-decay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--bn-freeze', default=True, type=bool)
    parser.add_argument('--scheduler-cosine', default=False, type=bool)
    args = parser.parse_args()
    # ----settings-----
    fp_config_out = '/tmp/config.py'
    if args.work_dir_name == '':
        work_dir_name = "lr_{}_wd_{}_momentum_{}".format(
            args.lr, args.weight_decay, args.momentum)
    else:
        work_dir_name = args.work_dir_name

    if args.bn_freeze:
        work_dir_name += "_bn_true"
    if args.scheduler_cosine:
        work_dir_name += "_cosine"

    print('work_dir_name:', work_dir_name)
    cfg = Config.fromfile(osp.join(args.dir_root, args.config))

    cfg_options = {
        'work_dir': osp.join(
            args.work_dir_root,
            work_dir_name),
        'data.train.ann_file': osp.join(
            args.train_file_dir,
            'breakfast_train_list_videos.txt'),
        'data.val.ann_file': osp.join(
            args.train_file_dir,
            'breakfast_val_list_videos.txt'),
        'data.test.ann_file': osp.join(
            args.train_file_dir,
            'breakfast_test_list_videos.txt'),
        'data.train.data_prefix': args.dir_videos_root,
        'data.val.data_prefix': args.dir_videos_root,
        'data.test.data_prefix': args.dir_videos_root,
        'load_from': args.load_from,
        'data_root': args.dir_videos_root,
        'data_root_val': args.dir_videos_root,
        'ann_file_train': osp.join(
            args.train_file_dir,
            'breakfast_train_list_videos.txt'),
        'ann_file_val': osp.join(
            args.train_file_dir,
            'breakfast_val_list_videos.txt'),
        'ann_file_test': osp.join(
            args.train_file_dir,
            'breakfast_test_list_videos.txt'),
        'data.videos_per_gpu': args.videos_per_gpu,
        'data.workers_per_gpu': args.workers_per_gpu,
        'optimizer.lr': args.lr,
        'optimizer.weight_decay': args.weight_decay,
        'optimizer.momentum': args.momentum,
        'model.backbone.norm_eval': args.bn_freeze,
        'total_epochs': args.epochs}
    if osp.exists(
        osp.join(
            args.work_dir_root,
            work_dir_name,
            'latest.pth')):
        cfg_options['resume_from'] = osp.join(
            args.work_dir_root,
            work_dir_name,
            'latest.pth')
    if args.scheduler_cosine:
        cfg_options['lr_config'] = dict(
            policy='CosineAnnealing',
            by_epoch=False,
            min_lr=0,
            warmup='linear',
            warmup_by_epoch=True,
            warmup_iters=1,
            warmup_ratio=0.1)
    else:
        cfg_options['lr_config'] = dict(policy='step', step=[20, 40])
    cfg.merge_from_dict(cfg_options)
    cfg.dump(fp_config_out)

    train_command = str(osp.join(args.dir_root, "tools/dist_train_onlyheader.sh")) + \
        " " + fp_config_out + " 1 --validate --seed 0 --deterministic --gpu-ids 0"
    import subprocess
    print(train_command)
    if not osp.exists(osp.join(
            osp.join(args.work_dir_root, work_dir_name),
            'epoch_50.pth')):
        os.system(train_command)
    if osp.exists(osp.join(
            osp.join(args.work_dir_root, work_dir_name),
            'epoch_50.pth')):
        test_command = "python " + str(
            osp.join(
                args.dir_root,
                "tools/test_several.py")) + " " + fp_config_out + " " + osp.join(
            osp.join(
                args.work_dir_root,
                work_dir_name),
            'epoch_50.pth') + " --eval top_k_accuracy mean_class_accuracy --out " + osp.join(
            osp.join(
                args.work_dir_root,
                work_dir_name),
            'test_result.json') + " --out-several " + osp.join(
            osp.join(
                args.work_dir_root,
                work_dir_name),
            'test_result_several.json')
        print(test_command)
        os.system(test_command)
