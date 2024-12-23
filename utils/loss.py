
import numpy as np
import torch.nn as nn
import torch
from utils.lovasz import lovasz_softmax
import torch.nn.functional as F

class criterion(nn.Module):
    def __init__(self, config):
        super(criterion, self).__init__()
        self.config = config
        self.lambda_lovasz = self.config['train_params'].get('lambda_lovasz', 0.1)
        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=seg_labelweights,
            ignore_index=config['dataset_params']['ignore_label']
        )
        self.lovasz_loss = Lovasz_loss(
            ignore=config['dataset_params']['ignore_label']
        )

    def forward(self, data_dict):
        loss_main_ce = self.ce_loss(data_dict['logits'], data_dict['labels'].long())
        loss_main_lovasz = self.lovasz_loss(F.softmax(data_dict['logits'], dim=1), data_dict['labels'].long())
        loss_main = loss_main_ce + loss_main_lovasz * self.lambda_lovasz
        data_dict['loss_main_ce'] = loss_main_ce
        data_dict['loss_main_lovasz'] = loss_main_lovasz
        data_dict['loss'] += loss_main

        return data_dict

class Lovasz_loss(nn.Module):
    def __init__(self, ignore=None):
        super(Lovasz_loss, self).__init__()
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, ignore=self.ignore)