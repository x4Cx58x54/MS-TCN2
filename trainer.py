import os
from datetime import datetime
import json
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
from loguru import logger

from configs import MSTCN2_Config
from model import MS_TCN2
from utils import read_nonempty_lines
import eval
import visualization

class Trainer:
    def __init__(self, config: MSTCN2_Config, class_weights=None):
        self.num_classes = len(config.actions_dict)
        self.model = MS_TCN2(config.num_layers_PG, config.num_layers_R, config.num_R, config.num_f_maps, config.features_dim, self.num_classes)

        self.dataset = config.dataset
        self.split = config.split
        self.class_weights = class_weights
        self.loss_lambda = config.loss_lambda

        self.config_name = config.name

    def train(self, config: MSTCN2_Config, batch_gen):
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join('logs', f'{config.name}{config.comment}_split{self.split}', time_now)
        os.makedirs(log_dir, exist_ok=True)
        logger.add(os.path.join(log_dir, 'loguru'))
        writer = SummaryWriter(log_dir)
        config_str = json.dumps(config.__dict__, ensure_ascii=False, indent=2)
        config_str = '  \n'.join(config_str.splitlines())
        writer.add_text('config', config_str)

        if self.class_weights is not None:
            weight = self.class_weights.to(config.device)
        else:
            weight = None
        self.ce = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
        self.mse = nn.MSELoss(reduction='none')

        self.model.train()
        self.model.to(config.device)
        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        for epoch in range(config.num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(config.batch_size)
                # [batchsize, dim=3072, n_clip], [batchsize, n_clip], [batchsize, n_class, n_clip]
                batch_input, batch_target, mask = batch_input.to(config.device), batch_target.to(config.device), mask.to(config.device)
                optimizer.zero_grad()
                predictions = self.model(batch_input) # [4, batchsize, n_classes, n_clip]

                loss = 0
                for p in predictions:
                    # print(F.log_softmax(p[:, :, 1:], dim=1).shape)
                    # print(F.log_softmax(p.detach()[:, :, :-1], dim=1).shape)
                    # exit(0)
                    loss += self.ce(
                        p.transpose(2, 1).contiguous().view(-1, self.num_classes), #[batchsize*n_clip, n_class]
                        batch_target.view(-1) # [batchsize*n_clip]
                    )
                    loss += self.loss_lambda*torch.mean(
                        torch.clamp(
                            self.mse(
                                F.log_softmax(p[:, :, 1:], dim=1),
                                F.log_softmax(p.detach()[:, :, :-1], dim=1)
                            ), min=0, max=16
                        )*mask[:, :, 1:]
                    )

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            log_loss = epoch_loss / len(batch_gen.list_of_examples)
            log_acc = float(correct)/total * 100
            logger.info("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, log_loss, log_acc))
            writer.add_scalar('Accuracy% / Train', log_acc, epoch+1)
            writer.add_scalar('Loss / Train', log_loss, epoch+1)

            if (epoch + 1) % config.checkpoint_freq == 0:
                torch.save(self.model.state_dict(), config.model_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), config.model_dir + "/epoch-" + str(epoch + 1) + ".opt")
            if (epoch + 1) % config.eval_freq == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.predict(config, config.vid_test_list_file, load=False)
                    cm_save_path, metrics_dict = eval.main(config)
                    writer.add_image('confusion matrix', pil_to_tensor(Image.open(cm_save_path)), epoch+1)
                    for k, v in metrics_dict.items():
                        writer.add_scalar(k, v, epoch+1)
                    for vid in config.visualization_video_list:
                        vis_path = visualization.main(config, vid, f' at epoch {epoch+1}')
                        # vis_fig = np.asarray(Image.open(vis_path))
                        writer.add_image(vid, pil_to_tensor(Image.open(vis_path)), epoch+1)
                    self.model.train()
            writer.flush()

    def predict(self, config: MSTCN2_Config, vid_list_file, load=True):
        self.model.eval()
        checkpoint_path = config.checkpoint_path_explicit
        with torch.no_grad():
            if load:
                if checkpoint_path is None:
                    self.model.load_state_dict(torch.load(config.model_dir + "/epoch-" + str(config.test_epoch) + ".model", map_location='cpu'))
                else:
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            self.model = self.model.to(config.device)
            list_of_vids = read_nonempty_lines(vid_list_file)
            for vid in list_of_vids:
                features_filename = os.path.join(config.features_path, os.path.splitext(vid)[0] + '.npy')
                features = np.load(features_filename)
                features = features[:, ::config.sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(config.device)
                predictions = self.model(input_x)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(config.actions_dict.keys())[list(config.actions_dict.values()).index(predicted[i].item())]]*config.sample_rate))
                with open(os.path.join(config.results_dir, os.path.splitext(vid)[0]), 'w') as f:
                    f.write('\n'.join(recognition))
