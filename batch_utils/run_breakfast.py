from re import A
from mmcv import Config, DictAction
from numpy import argsort
from mmaction.apis import inference_recognizer, init_recognizer
import os.path as osp
import os
import time
from glob import glob
import argparse
import numpy as np

# example
# python /tmp/repo/batch_utils/run_breakfast.py --lr 0.01 --bn-freeze 0 --scheduler-cosine 1 --workers-per-gpu 15
# python /tmp/repo/batch_utils/run_breakfast.py --work-dir-name --lr 0.01
# --bn-freeze 1 --scheduler-cosine 1 --workers-per-gpu 15 --load-from
# /lfovision_log/tsm_learningrate/lr_0.01_wd_0.0005_momentum_0.9_bn_true_cosine
# --work-dir-root /lfovision_log/tsm_after_manual_correction/
# --train-file-path
# /lfovision_sthv2_breakfast/annotations/experiment_tsm_after_manual_correction/iteration_1_after_manualcheck/breakfast_train_list_videos_mixed.txt
# --val-file-path
# /lfovision_sthv2_breakfast/annotations/experiment_tsm_after_manual_correction/breakfast_val_list_videos.txt
# --test-file-path
# /lfovision_sthv2_breakfast/annotations/experiment_tsm_after_manual_correction/breakfast_test_list_videos.txt
# python3 launch.py --cmd "python /tmp/repo/batch_utils/run_breakfast.py --lr 0.0001 # --bn-freeze 1 --scheduler-cosine 1 --videos-per-gpu 6 --workers-per-gpu 4 # --work-dir-root /lfovision_log/tsm_learningrate_alllayers_considering_labelbias/ # --only-header 0 --base-frozen-stages 1 --modify-class-bias 1"
# git pull && python /tmp/repo/batch_utils/run_breakfast.py --lr 0.0001
# --bn-freeze 1 --scheduler-cosine 1 --videos-per-gpu 6 --workers-per-gpu
# 4 --work-dir-root
# /lfovision_log/tsm_learningrate_alllayers_considering_labelbias_augmetation_flip/
# --only-header 0 --base-frozen-stages 1 --modify-class-bias 1 --debug 1
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
        '--train-file-path',
        default='',
        type=str)
    parser.add_argument(
        '--test-file-path',
        default='',
        type=str)
    parser.add_argument(
        '--val-file-path',
        default='',
        type=str)
    parser.add_argument(
        '--dir-videos-root',
        default='/lfovision_sthv2_breakfast/',
        type=str)
    parser.add_argument('--videos-per-gpu', default=8, type=int)
    parser.add_argument('--workers-per-gpu', default=6, type=int)
    parser.add_argument('--lr', default=0.0075, type=float)
    parser.add_argument('--weight-decay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--bn-freeze', default=1, type=int)
    parser.add_argument('--scheduler-cosine', default=0, type=int)
    parser.add_argument('--only-header', default=1, type=int)
    parser.add_argument('--base-frozen-stages', default=-1, type=int)
    parser.add_argument('--modify-class-bias', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--color', default=0, type=int)
    parser.add_argument('--distributed', default=0, type=int)
    args = parser.parse_args()
    # ----settings-----
    if len(args.train_file_path) == 0:
        train_file_path = osp.join(
            args.train_file_dir,
            'breakfast_train_list_videos.txt')
    else:
        train_file_path = args.train_file_path

    if len(args.val_file_path) == 0:
        val_file_path = osp.join(
            args.train_file_dir,
            'breakfast_val_list_videos.txt')
    else:
        val_file_path = args.val_file_path

    if len(args.test_file_path) == 0:
        test_file_path = osp.join(
            args.train_file_dir,
            'breakfast_test_list_videos.txt')
    else:
        test_file_path = args.test_file_path

    fp_config_out = '/tmp/config.py'
    if args.work_dir_name == '':
        work_dir_name = "lr_{}_wd_{}_momentum_{}".format(
            args.lr, args.weight_decay, args.momentum)
    else:
        work_dir_name = args.work_dir_name

    if args.bn_freeze == 1:
        work_dir_name += "_bn_true"
        bool_bn_freeze = True
    else:
        bool_bn_freeze = False
    if args.scheduler_cosine == 1:
        work_dir_name += "_cosine"

    print('work_dir_name:', work_dir_name)
    cfg = Config.fromfile(osp.join(args.dir_root, args.config))
    cfg_options = {
        'work_dir': osp.join(
            args.work_dir_root,
            work_dir_name),
        'data.train.ann_file': train_file_path,
        'data.val.ann_file': val_file_path,
        'data.test.ann_file': test_file_path,
        'data.train.data_prefix': args.dir_videos_root,
        'data.val.data_prefix': args.dir_videos_root,
        'data.test.data_prefix': args.dir_videos_root,
        'load_from': args.load_from,
        'data_root': args.dir_videos_root,
        'data_root_val': args.dir_videos_root,
        'ann_file_train': train_file_path,
        'ann_file_val': val_file_path,
        'ann_file_test': test_file_path,
        'data.videos_per_gpu': args.videos_per_gpu,
        'data.workers_per_gpu': args.workers_per_gpu,
        'optimizer.lr': args.lr,
        'optimizer.weight_decay': args.weight_decay,
        'optimizer.momentum': args.momentum,
        'model.backbone.norm_eval': bool_bn_freeze,
        'model.backbone.frozen_stages': args.base_frozen_stages,
        # frozen_stages (int): Stages to be frozen (all param fixed). -1 means
        # not freezing any parameters. Default: -1.
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
    if args.scheduler_cosine == 1:
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
    if args.modify_class_bias == 1:
        with open(train_file_path, 'r') as f:
            lines = f.readlines()
        labels = [int(item.split(' ')[1].strip()) for item in lines]
        # print(labels)
        class_num = len(list(set(labels)))
        label_count = [labels.count(i) for i in range(class_num)]
        cfg.model.cls_head.class_bias = label_count
        cfg.optimizer.constructor = 'TSMOptimizerConstructor_WO_BIAS'
    else:
        cfg.model.cls_head.class_bias = []
    if args.flip == 1:
        #import pdb
        # pdb.set_trace()
        cfg.data.train.pipeline = [
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=8),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
            dict(type='Flip', flip_ratio=0.5, lazy=True),
            dict(
                type='Normalize', mean=[
                    123.675, 116.28, 103.53], std=[
                    58.395, 57.12, 57.375], to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
        cfg.train_pipeline = cfg.data.train.pipeline
    if args.color == 1 and args.flip == 1:
        cfg.data.train.pipeline = [
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=8),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
            dict(type='VideoAug', degrees = 10, prob = 0.01),       # use a custom pipeline
            dict(type='Flip', flip_ratio=0.5, lazy=True),
            dict(
                type='Normalize', mean=[
                    123.675, 116.28, 103.53], std=[
                    58.395, 57.12, 57.375], to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
        # import pdb; pdb.set_trace()
        cfg.train_pipeline = cfg.data.train.pipeline
    cfg.merge_from_dict(cfg_options)
    cfg.dump(fp_config_out)
    if args.distributed == 0:
        if args.only_header == 1:
            # train_command = str(osp.join(args.dir_root, "tools/dist_train_onlyheader.sh")) + \
            #    " " + fp_config_out + " 1 --validate --seed 0 --deterministic --gpu-ids 0"
            train_command = "python " + str(osp.join(args.dir_root, "tools/train_onlyheader.py")) + \
                " " + fp_config_out + " --validate --seed 0 --deterministic --gpu-ids 0"
        else:
            train_command = "python " + str(osp.join(args.dir_root, "tools/train.py")) + \
                " " + fp_config_out + " --validate --seed 0 --deterministic --gpu-ids 0"
        import subprocess
    else:
        train_command = str(osp.join(args.dir_root, "tools/dist_train.sh")) + \
            " " + fp_config_out + " " + int(args.distributed) + " --validate --seed 0 --deterministic"        
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
