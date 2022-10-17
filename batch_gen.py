import os
import random

import numpy as np
import torch

from configs import MSTCN2_Config
from utils import read_nonempty_lines


class BatchGenerator:
    def __init__(self, config: MSTCN2_Config, vid_list_file):
        self.list_of_examples = read_nonempty_lines(vid_list_file)
        # random.shuffle(self.list_of_examples)
        self.index = 0
        self.actions_dict = config.actions_dict
        self.num_classes = len(self.actions_dict)
        self.gt_path = config.gt_path
        self.features_path = config.features_path
        self.sample_rate = config.sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            feature_file_path = os.path.join(self.features_path, os.path.splitext(vid)[0]+'.npy')
            features = np.load(feature_file_path)
            gt_file_path = os.path.join(self.gt_path, vid)
            gt_content = read_nonempty_lines(gt_file_path)
            classes = np.zeros(min(np.shape(features)[1], len(gt_content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[gt_content[i]]
            batch_input .append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
