# dataset settings
dataset_type = 'ActivityNetDataset'
data_root = '/dataset/maction_feat_with_label/'#'/dataset/activitynet/maction_feat/'
data_root_val = '/dataset/maction_feat_with_label/'
ann_file_train = '/dataset/annotation/anet_anno_train.json'
ann_file_val = '/dataset/annotation/anet_anno_test.json'
ann_file_test = '/dataset/annotation/anet_anno_test.json'


work_dir = 'work_dirs/bsn_800x100_20e_1x16_activitynet_feature_train_with_label/'
tem_results_dir = f'{work_dir}/tem_results/'
pgm_proposals_dir = f'{work_dir}/pgm_proposals/'
pgm_features_dir = f'{work_dir}/pgm_features/'

temporal_scale = 100
pgm_proposals_cfg = dict(
    pgm_proposals_thread=8, temporal_scale=temporal_scale, peak_threshold=0.5)
pgm_features_test_cfg = dict(
    pgm_features_thread=4,
    top_k=150,#1000,
    num_sample_start=8,
    num_sample_end=8,
    num_sample_action=16,
    num_sample_interp=3,
    bsp_boundary_ratio=0.2)
pgm_features_train_cfg = dict(
    pgm_features_thread=4,
    top_k=150,#500,
    num_sample_start=8,
    num_sample_end=8,
    num_sample_action=16,
    num_sample_interp=3,
    bsp_boundary_ratio=0.2)
