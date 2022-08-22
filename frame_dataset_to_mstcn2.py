import os
import pickle
from typing import Literal

import numpy as np
from tqdm import tqdm


target_path = 'data/ktd_1fps_3072'

dataset_path = 'path_to_dataset'
videomae_feature_path = os.path.join(dataset_path, 'features_videomae_320x320_1fps')
assert os.path.isdir(videomae_feature_path), videomae_feature_path

target_feature_path = os.path.join(target_path, 'features_videomae_320x320_1fps')
target_gt_path = os.path.join(target_path, 'groundTruth')
target_splits_path = os.path.join(target_path, 'splits')

os.mkdir(target_path)
os.mkdir(target_feature_path)
os.mkdir(target_gt_path)
os.mkdir(target_splits_path)

target_mapping_path = os.path.join(target_path, 'mapping.txt')

with open(os.path.join(dataset_path, 'fused_annotation.pkl'), 'rb') as f:
    annotations = pickle.load(f)

with open(target_mapping_path, 'w') as f:
    action_dict = annotations['meta_data']['action_type_names']
    for action_id, action_name in action_dict.items():
        f.write(f'{action_id} {action_name}\n')

def merge_multiview_features(
    dim_output: int,
    mode: Literal['concat', 'avg'],
    feat_f_path: str,
    feat_s_path: str,
    feat_t_path: str):
    with open(feat_f_path, 'rb') as f:
        feat_f = pickle.load(f)
    with open(feat_s_path, 'rb') as f:
        feat_s = pickle.load(f)
    with open(feat_t_path, 'rb') as f:
        feat_t = pickle.load(f)
    assert feat_f.shape[1] == feat_s.shape[1] == feat_t.shape[1]
    dim_input = feat_f.shape[1]
    effective_frame_num = min([feat_f.shape[0], feat_s.shape[0], feat_t.shape[0]])
    slice_and_T = lambda x: x.detach().cpu().numpy()[:effective_frame_num, :].T
    feat_f = slice_and_T(feat_f)
    feat_s = slice_and_T(feat_s)
    feat_t = slice_and_T(feat_t)
    if mode == 'concat':
        res = np.concatenate((feat_f, feat_s, feat_t))
        assert dim_output == res.shape[0], 'Set dim_output manually is still required.'
    elif mode == 'avg':
        res = (feat_f + feat_s + feat_t) / 3
    assert res.shape == (dim_output, effective_frame_num)
    return res

gt_sample_name_list = []
basename = lambda x: os.path.splitext(x)[0]
print('Processing video features...')
for vid_anno in tqdm(annotations['annotations']):
    vid_name = vid_anno['video_front_view']
    assert vid_name[5:7] == '_f', vid_name
    sample_name = vid_name[:5]

    feat_path = lambda x: basename(os.path.join(videomae_feature_path, vid_anno[x])) + '.pkl'
    feat_f_path = feat_path('video_front_view')
    assert os.path.isfile(feat_f_path), feat_f_path
    feat_s_path = feat_path('video_side_view')
    assert os.path.isfile(feat_s_path), feat_s_path
    feat_t_path = feat_path('video_top_view')
    assert os.path.isfile(feat_t_path), feat_t_path

    gt_sample_name_list.append(sample_name + '.txt')
    res_gt_path = os.path.join(target_gt_path, sample_name + '.txt')
    res_feature_path = os.path.join(target_feature_path, sample_name + '.npy')

    merged_feat = merge_multiview_features(3072, 'concat', feat_f_path, feat_s_path, feat_t_path)
    # merged_feat = merge_multiview_features(1024, 'avg', feat_f_path, feat_s_path, feat_t_path)
    np.save(res_feature_path, merged_feat)
    dim_output, effective_frame_num = merged_feat.shape

    # for frame_id in range(effective_frame_num):
    gt_list = [0] * effective_frame_num
    for action in vid_anno['annotation_for_video']:
        # action_type start_sec end_sec score
        action_type = action['action_type']
        start_sec = action['start_sec']
        end_sec = action['end_sec']
        score = action['score']
        for i in range(start_sec, end_sec+1):
            if i < effective_frame_num:
                gt_list[i] = action_dict[action_type]
    for i in range(effective_frame_num):
        if gt_list[i] == 0:
            gt_list[i] = 'background'
    with open(res_gt_path, 'w') as f:
        for i in range(effective_frame_num):
            f.write(gt_list[i] + '\n')
print('Processing video features done.')


def generate_splits(train_sample_num, split_id):
    # gt_sample_name_list = sorted(gt_sample_name_list)
    train_split_path = os.path.join(target_splits_path, f'train.split{split_id}.bundle')
    test_split_path = os.path.join(target_splits_path, f'test.split{split_id}.bundle')
    print(f'generate_splits: ')
    print(train_split_path)
    print(test_split_path)
    with open(train_split_path, 'w') as f:
        for name in gt_sample_name_list[:train_sample_num]:
            f.write(name + '\n')
    with open(test_split_path, 'w') as f:
        for name in gt_sample_name_list[train_sample_num:]:
            f.write(name + '\n')
    print(f'generate_splits done.')

generate_splits(140, 0)
print(f'Data saved to {target_path}')
