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
parser.add_argument('--action', type=str, choices=['train', 'predict'], default='train')
parser.add_argument('--weights', type=str, default='default')
args = parser.parse_args()
config = configs[args.config_name]

device = torch.device(config.device)
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

model_dir = config.model_dir
results_dir = config.results_dir

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

for dir in (model_dir, results_dir):
    with open(os.path.join(dir, 'config'), 'w') as f:
        f.write(str(config))

durations = [7294, 1089, 3055, 16583, 6583, 2544, 2023, 4684, 1766, 3071, 3056, 5491, 1309, 13140, 3628, 7879, 7871, 1006, 992, 1068, 2433, 628, 1499, 11834, 827, 2100, 1793, 2406, 12172, 8204, 4108, 36637, 149766]

weights = 1/torch.tensor(durations)
weights = weights / torch.sum(weights)
if args.weights == 'None':
    weights = None
    print('None weights')

trainer = Trainer(config, class_weights=weights)
if args.action == "train":
    batch_gen = BatchGenerator(config, config.vid_train_list_file)
    trainer.train(config, batch_gen)

if args.action == "predict":
    trainer.predict(config, config.vid_test_list_file)
