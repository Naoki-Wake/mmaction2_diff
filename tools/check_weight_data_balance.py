import numpy as np
import torch
from mmcv import Config
from mmaction.apis import inference_recognizer, init_recognizer
import os.path as osp
import os
import argparse
import matplotlib.pyplot as plt
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import mmaction_diff.models.heads.household_head_nonaddlayer
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run breakfast')
    parser.add_argument('--dir-root', default='/tmp/repo', type=str)
    parser.add_argument(
        '--checkpointdir',
        default='/lfovision_log/tsm_learningrate/lr_0.01_wd_0.0005_momentum_0.9_bn_true_cosine',
        type=str)
    parser.add_argument(
        '--annotationdir',
        default='/lfovision_sthv2_breakfast/annotations/with_pseudo_largedatanum/',
        type=str)
    parser.add_argument(
        '--outpathdir',
        default='/lfovision_log/debug/check_weight_data_balance/',
        type=str)

    args = parser.parse_args()
    #----settings-----
    checkpoint = osp.join(args.checkpointdir, 'epoch_50.pth')
    fp_config = osp.join(args.checkpointdir, 'config.py')
    dir_out = args.outpathdir
    # assign the desired device.
    device = torch.device('cuda:0')
    cfg = Config.fromfile(fp_config)
    cfg_options = {}
    cfg.merge_from_dict(cfg_options)
    # build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(fp_config, checkpoint, device=device)
    weight = model.cls_head.fc_cls.weight.cpu().detach().numpy()
    bias = model.cls_head.fc_cls.bias.cpu().detach().numpy()

    filename = osp.join(args.annotationdir, 'breakfast_train_list_videos.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()

    labels = [int(item.split(' ')[1].strip()) for item in lines]
    #print(labels)
    class_num = len(list(set(labels)))
    label_count = [labels.count(i) for i in range(class_num)]
    label_count = np.array(label_count)
    # normalize
    label_count = label_count/np.max(label_count)
    # load data
    weights = np.sum(weight,axis=1)
    #weights = [np.linalg.norm(weight[row,:], ord=2) for row in range(11)]
    # normalize
    weights = weights/np.max(weights)
    if os.path.exists(dir_out) is False:
        os.makedirs(dir_out)
    # scatter plot
    plt.scatter(label_count,weights)
    plt.xlabel('label ratio')
    plt.ylabel('weight')
    plt.grid(True)
    # save figure
    #plt.show()
    plt.savefig(str(osp.join(dir_out,'label_weight.png')))
    # clear figure
    plt.clf()

    bias = bias/np.max(bias)
    plt.scatter(label_count,bias)
    plt.xlabel('label ratio')
    plt.ylabel('bias')
    plt.grid(True)
    # save figure
    #plt.show()
    plt.savefig(osp.join(dir_out,'label_bias.png'))
    # clear figure
    plt.clf()
    #copy this python file to /lfovision_log/debug/check_weight_data_balance
    import shutil
    shutil.copy('./check_weight_data_balance.py',osp(dir_out,'check_weight_data_balance.py'))
    