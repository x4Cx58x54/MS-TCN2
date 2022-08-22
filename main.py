import os
import argparse
import random

import torch

from trainer import Trainer
from batch_gen import BatchGenerator
from configs import configs
from utils import read_nonempty_lines

parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str)
parser.add_argument('--action', type=str, choices=['train', 'predict'])
args = parser.parse_args()
config = configs[args.config_name]

device = torch.device(config.device)
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if config.dataset == "50salads":
    sample_rate = 2

dataset_path = os.path.join('data', config.dataset)
features_path = os.path.join(dataset_path, config.feature_dir)
gt_path = os.path.join(dataset_path, config.gt_dir)
splits_path = os.path.join(dataset_path, config.splits_dir)
vid_train_list_file = os.path.join(splits_path, f'train.split{config.split}.bundle')
vid_test_list_file = os.path.join(splits_path, f'test.split{config.split}.bundle')
mapping_file = os.path.join(dataset_path, 'mapping.txt')

model_dir = os.path.join('models', config.dataset, f'split_{config.split}')
results_dir = os.path.join('results', config.dataset, f'split_{config.split}')

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

actions = read_nonempty_lines(mapping_file)
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)
trainer = Trainer(config.num_layers_PG, config.num_layers_R, config.num_R, config.num_f_maps, config.features_dim, num_classes, config.dataset, config.split)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_train_list_file)
    trainer.train(model_dir, batch_gen, num_epochs=config.num_epochs, batch_size=config.batch_size, learning_rate=config.lr, device=device)

if args.action == "predict":
    trainer.predict(
        model_dir, results_dir, features_path, vid_test_list_file,
        config.test_epoch, actions_dict, device, sample_rate,
        config.checkpoint_path_explicit
    )
