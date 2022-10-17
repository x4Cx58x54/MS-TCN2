import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from utils import read_nonempty_lines

@dataclass
class MSTCN2_Config:
    name: str
    comment: str = ''
    # General configs
    device: str = 'cuda:8'
    # Data configs
    dataset_root_dir: str = 'data'
    dataset: str = 'gtea'
    split: str = '1'
    feature_dir: str = 'features'
    gt_dir: str = 'groundTruth'
    splits_dir: str = 'splits'
    # Model hyperparameters
    features_dim: int = 2048
    num_layers_PG: int = 11
    num_layers_R: int = 10
    num_R: int = 3
    loss_lambda: float = 0.15
    # Training configs
    num_epochs: int = 100
    batch_size: int = 16
    lr: float = 0.0005
    num_f_maps: int = 64
    # Eval configs
    test_epoch: int = 100
    eval_freq: int = 10
    visualization_video_list: Tuple[str] = tuple()
    checkpoint_freq: int = 50
    checkpoint_path_explicit: Optional[str] = None # explicitly specify checkpoint path, optional

    def __post_init__(self):
        # use the full temporal resolution @ 15fps
        self.sample_rate = 1
        # sample input features @ 15fps instead of 30 fps
        # for 50salads, and up-sample the output to 30 fps
        if self.dataset == "50salads":
            self.sample_rate = 2

        self.dataset_path = os.path.join(self.dataset_root_dir, self.dataset)
        self.features_path = os.path.join(self.dataset_path, self.feature_dir)
        self.gt_path = os.path.join(self.dataset_path, self.gt_dir)
        self.splits_path = os.path.join(self.dataset_path, self.splits_dir)
        self.vid_train_list_file = os.path.join(self.splits_path, f'train.split{self.split}.bundle')
        self.vid_test_list_file = os.path.join(self.splits_path, f'test.split{self.split}.bundle')
        self.mapping_file = os.path.join(self.dataset_path, 'mapping.txt')

        self.model_dir = os.path.join('models', self.name, f'split_{self.split}')
        self.results_dir = os.path.join('results', self.name, f'split_{self.split}')

        actions = read_nonempty_lines(self.mapping_file)
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])


configs: Dict[str, MSTCN2_Config] = {
    'default': MSTCN2_Config('default'),
    'gtea': MSTCN2_Config(
        'gtea',
        device='cuda:8',
        split='1',
        dataset='gtea',
        batch_size=16,
        num_epochs=300,
        checkpoint_freq=100,
        eval_freq=10,
    ),
    'breakfast': MSTCN2_Config(
        'breakfast',
        device='cuda:8',
        split='1',
        dataset='breakfast',
        batch_size=16,
        num_epochs=200,
        checkpoint_freq=100,
        eval_freq=10,
    ),
    'ktd_5fps_3072_from25_fmaps64_PG14_lR15_R4': MSTCN2_Config(
        'ktd_5fps_3072_from25_fmaps64_PG14_lR15_R4',
        device='cuda:8',
        split='0',
        dataset='ktd_5fps_3072_from25',
        feature_dir='features_videomae_25fps',
        # gt_dir='gt_zyh',
        num_layers_PG=14,
        num_layers_R=15,
        num_R=4,
        # loss_lambda=0.15,
        features_dim=3072,
        batch_size=4,
        num_f_maps=64,
        num_epochs=400,
        # test_epoch=50,
        checkpoint_freq=50,
        eval_freq=10,
        visualization_video_list=('047_0', '047_1', '047_2', '048_0'),
    ),
    'ktd_5fps_3072_from25_fmaps64_PG13_lR11_R3': MSTCN2_Config(
        'ktd_5fps_3072_from25_fmaps64_PG13_lR11_R3',
        device='cuda:8',
        split='0',
        dataset='ktd_5fps_3072_from25',
        feature_dir='features_videomae_25fps',
        # gt_dir='gt_zyh',
        num_layers_PG=13,
        num_layers_R=11,
        num_R=3,
        # loss_lambda=0.15,
        features_dim=3072,
        batch_size=8,
        num_f_maps=64,
        num_epochs=400,
        # test_epoch=50,
        checkpoint_freq=50,
        eval_freq=10,
        visualization_video_list=('047_0', '047_1', '047_2', '048_0'),
    ),
    'ktd_5fps_3072_from25_fmaps64_all_train': MSTCN2_Config(
        'ktd_5fps_3072_from25_fmaps64_all_train',
        device='cuda:8',
        split='0',
        dataset='ktd_5fps_3072_from25',
        feature_dir='features_videomae_25fps',
        features_dim=3072,
        batch_size=8,
        num_f_maps=64,
        num_epochs=400,
        # test_epoch=50,
        checkpoint_freq=50,
        eval_freq=10,
        visualization_video_list=('047_0', '047_1', '047_2', '048_0'),
    ),
    'ktd_5fps_3072_from25_fmaps96_all_train': MSTCN2_Config(
        'ktd_5fps_3072_from25_fmaps96_all_train',
        device='cuda:9',
        split='0',
        dataset='ktd_5fps_3072_from25',
        feature_dir='features_videomae_25fps',
        # gt_dir='gt_zyh',
        # num_layers_PG=15,
        # num_layers_R=10,
        # num_R=3,
        # loss_lambda=0.15,
        features_dim=3072,
        batch_size=4,
        num_f_maps=96,
        num_epochs=400,
        # test_epoch=50,
        checkpoint_freq=50,
        eval_freq=10,
        visualization_video_list=('047_0', '047_1', '047_2', '048_0'),
    ),
    'ktd_5fps_3072_from25_fmap64': MSTCN2_Config(
        'ktd_5fps_3072_from25_fmap64',
        device='cuda:8',
        split='1',
        dataset='ktd_5fps_3072_from25',
        feature_dir='features_videomae_25fps',
        # gt_dir='gt_zyh',
        # num_layers_PG=11,
        # num_layers_R=10,
        # num_R=3,
        # loss_lambda=0.15,
        features_dim=3072,
        batch_size=8,
        num_f_maps=64,
        num_epochs=500,
        # test_epoch=50,
        checkpoint_freq=50,
        eval_freq=10,
        visualization_video_list=('047_0', '047_1', '047_2', '048_0'),
    ),
    'ktd_5fps_3072_from25_fmap96': MSTCN2_Config(
        'ktd_5fps_3072_from25_fmap96',
        device='cuda:9',
        split='1',
        dataset='ktd_5fps_3072_from25',
        feature_dir='features_videomae_25fps',
        # gt_dir='gt_zyh',
        # num_layers_PG=11,
        # num_layers_R=10,
        # num_R=3,
        # loss_lambda=0.15,
        features_dim=3072,
        batch_size=4,
        num_f_maps=96,
        num_epochs=500,
        # test_epoch=50,
        checkpoint_freq=50,
        eval_freq=10,
        visualization_video_list=('047_0', '047_1', '047_2', '048_0'),
    ),
    'ktd_1fps_i3d_4096': MSTCN2_Config(
        'ktd_1fps_i3d_4096',
        device='cuda:9',
        split='0',
        dataset='ktd_1fps_4096',
        feature_dir='features_i3d_1fps',
        # gt_dir='gt_zyh',
        # num_layers_PG=11,
        # num_layers_R=10,
        # num_R=3,
        # loss_lambda=0.15,
        features_dim=4096,
        batch_size=16,
        num_f_maps=128,
        num_epochs=300,
        # test_epoch=50,
        checkpoint_freq=50,
        eval_freq=10,
        visualization_video_list=('047_0', '047_1', '047_2', '048_0'),
    ),
}


if __name__ == '__main__':
    import json
    config = configs['ktd_5fps_3072_from25'].__dict__
    config_str = json.dumps(config, ensure_ascii=False, indent=2)
    print(configs['ktd_5fps_3072_from25'].actions_dict)
    print(config_str)
