import os
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
from loguru import logger

from model import MS_TCN2
from utils import read_nonempty_lines


class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split):
        self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

        self.dataset = dataset
        self.split = split

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join('logs', f'{self.dataset}_split{self.split}', time_now)
        os.makedirs(log_dir, exist_ok=True)
        logger.add(os.path.join(log_dir, 'loguru'))
        writer = SummaryWriter(log_dir)

        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            log_loss = epoch_loss / len(batch_gen.list_of_examples)
            log_acc = float(correct)/total
            logger.info("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, log_loss, log_acc))
            writer.add_scalar('train acc', log_acc, epoch+1)
            writer.add_scalar('train loss', log_loss, epoch+1)

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate, checkpoint_path):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            if checkpoint_path is None:
                self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            else:
                self.model.load_state_dict(torch.load(checkpoint_path))
            list_of_vids = read_nonempty_lines(vid_list_file)
            for vid in list_of_vids:
                features_filename = os.path.join(features_path, os.path.splitext(vid)[0] + '.npy')
                features = np.load(features_filename)
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                with open(os.path.join(results_dir, os.path.splitext(vid)[0]), 'w') as f:
                    f.write('\n'.join(recognition))
