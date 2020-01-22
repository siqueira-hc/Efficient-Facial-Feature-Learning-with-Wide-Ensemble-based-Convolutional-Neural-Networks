#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO: Write docstring
TODO: Refactor: clean and add comments
"""

__author__ = "Henrique Siqueira"
__email__ = "siqueira.hc@outlook.com"
__license__ = "MIT license"
__version__ = "0.1"

# Standard libraries
from os import path, makedirs
import copy

# External libraries
import torch.nn.functional as F
import torch.nn as nn
import torch


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

    @staticmethod
    def get_num_layers():
        return 4

    def forward(self, x_base_to_process):
        x_base = F.relu(self.bn1(self.conv1(x_base_to_process)))
        x_base = self.pool(F.relu(self.bn2(self.conv2(x_base))))

        x_base = F.relu(self.bn3(self.conv3(x_base)))
        x_base = self.pool(F.relu(self.bn4(self.conv4(x_base))))

        return x_base


class BranchCategorical(nn.Module):
    def __init__(self):
        super(BranchCategorical, self).__init__()

        self.conv1 = nn.Conv2d(128, 128, 3, 1)
        self.conv2 = nn.Conv2d(128, 256, 3, 1)

        self.conv3 = nn.Conv2d(256, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)

        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, 8)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.mask = torch.ones((1, 1, 21, 21))

    @staticmethod
    def get_num_layers():
        return 5

    def forward(self, x_branch_to_process):
        x_branch = F.relu(self.bn1(self.conv1(x_branch_to_process)))
        x_branch = self.pool(F.relu(self.bn2(self.conv2(x_branch))))

        x_branch = F.relu(self.bn3(self.conv3(x_branch)))
        x_branch = self.global_pool(F.relu(self.bn4(self.conv4(x_branch))))

        x_branch = x_branch.view(-1, 512)
        x_branch = self.fc(x_branch)

        return x_branch

    def set_mask(self, mask_grad):
        self.mask = nn.functional.interpolate(mask_grad, scale_factor=3, mode="bilinear", align_corners=True)

    def forward_to_last_conv_layer(self, x_forward_to_last_conv_layer):
        x_grad_to = F.relu(self.bn1(self.conv1(x_forward_to_last_conv_layer)))
        x_grad_to = self.pool(F.relu(self.bn2(self.conv2(x_grad_to))))

        x_grad_to = F.relu(self.bn3(self.conv3(x_grad_to)))
        x_grad_to = F.relu(self.bn4(self.conv4(x_grad_to)))

        return x_grad_to

    def forward_from_last_conv_layer(self, x_forward_from_last_conv_layer):
        x_grad_from = self.global_pool(x_forward_from_last_conv_layer)
        x_grad_from = x_grad_from.view(-1, 512)
        x_grad_from = self.fc(x_grad_from)

        return x_grad_from

    @staticmethod
    def get_mask_size():
        return 6


# Branch Cat <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class Branch(nn.Module):
    def __init__(self):
        super(Branch, self).__init__()

        self.conv1 = nn.Conv2d(128, 128, 3, 1)
        self.conv2 = nn.Conv2d(128, 256, 3, 1)

        self.conv3 = nn.Conv2d(256, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)

        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, 8)

        self.fc_dimensional = nn.Linear(8, 2)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.mask = torch.ones((1, 1, 21, 21))

    @staticmethod
    def get_num_layers():
        return 6

    def forward(self, x_branch_to_process):
        x_branch = F.relu(self.bn1(self.conv1(x_branch_to_process)))
        x_branch = self.pool(F.relu(self.bn2(self.conv2(x_branch))))

        x_branch = F.relu(self.bn3(self.conv3(x_branch)))
        x_branch = self.global_pool(F.relu(self.bn4(self.conv4(x_branch))))

        x_branch = x_branch.view(-1, 512)

        emo = self.fc(x_branch)

        x_branch = F.relu(emo)

        x_branch = self.fc_dimensional(x_branch)

        return emo, x_branch

    def set_mask(self, mask_grad):
        self.mask = nn.functional.interpolate(mask_grad, scale_factor=3, mode="bilinear", align_corners=True)

    def forward_to_last_conv_layer(self, x_forward_to_last_conv_layer):
        x_grad_to = F.relu(self.bn1(self.conv1(x_forward_to_last_conv_layer)))
        x_grad_to = self.pool(F.relu(self.bn2(self.conv2(x_grad_to))))

        x_grad_to = F.relu(self.bn3(self.conv3(x_grad_to)))
        x_grad_to = F.relu(self.bn4(self.conv4(x_grad_to)))

        return x_grad_to

    def forward_from_last_conv_layer(self, x_forward_from_last_conv_layer):
        x_grad_from = self.global_pool(x_forward_from_last_conv_layer)
        x_grad_from = x_grad_from.view(-1, 512)
        x_grad_from = self.fc(x_grad_from)

        return x_grad_from

    @staticmethod
    def get_mask_size():
        return 6


class Ensemble(nn.Module):

    INPUT_IMAGE_SIZE = (96, 96)
    INPUT_IMAGE_NORMALIZATION_MEAN = [0.0, 0.0, 0.0]
    INPUT_IMAGE_NORMALIZATION_STD = [1.0, 1.0, 1.0]

    def __init__(self):
        super(Ensemble, self).__init__()

        self.base = Base()
        self.branches = []
        self.branches.append(Branch())

    def get_num_layers(self):
        return Base.get_num_layers() + (Branch.get_num_layers() * self.get_ensemble_size())

    def get_ensemble_size(self):
        return len(self.branches)

    def add_branch(self):
        self.branches.append(Branch())

    def forward(self, x):
        x_ensemble = self.base(x)

        emotion = []
        affect = []
        for branch in self.branches:
            e, a = branch(x_ensemble)
            emotion.append(e)
            affect.append(a)

        return emotion, affect

    @staticmethod
    def save(state_dicts, base_path_to_save_model, is_branch):
        token = "B-" if is_branch else ""

        if not path.isdir(path.join(base_path_to_save_model, str(len(state_dicts) - 1))):
            makedirs(path.join(base_path_to_save_model, str(len(state_dicts) - 1)))

        torch.save(state_dicts[0], path.join(base_path_to_save_model, str(len(state_dicts) - 1), "{}Net-Base-Shared_Representations.pkl".format(token)))

        for i in range(1, len(state_dicts)):
            torch.save(state_dicts[i], path.join(base_path_to_save_model, str(len(state_dicts) - 1), "{}Net-Branch_{}.pkl".format(token, i)))

        print("Network has been successfully saved at: {}".format(path.join(base_path_to_save_model, str(len(state_dicts) - 1))))

    @staticmethod
    def load(device):
        ensemble_size = 9
        loaded_model = Ensemble()
        loaded_model.branches = []

        # Load Base
        loaded_model_base = Base()
        loaded_model_base.load_state_dict(torch.load(path.join("./model/esr/trained_models/esr_9", "Net-Base-Shared_Representations.pkl"), map_location=device))
        loaded_model.base = loaded_model_base

        # Load Branches
        for i in range(1, ensemble_size + 1):
            loaded_model_branch = Branch()
            loaded_model_branch.load_state_dict(torch.load(path.join("./model/esr/trained_models/esr_9", "Net-Branch_{}.pkl".format(i)), map_location=device))

            loaded_model.branches.append(loaded_model_branch)

        return loaded_model

    def to_state_dict(self):
        state_dicts = [copy.deepcopy(self.base.state_dict())]

        for b in self.branches:
            state_dicts.append(copy.deepcopy(b.state_dict()))

        return state_dicts

    # TODO: model = nn.DataParallel(model)
    def to_device(self, device_to_process="cpu"):
        self.to(device_to_process)
        self.base.to(device_to_process)

        for b_td in self.branches:
            b_td.to(device_to_process)
            b_td.mask = b_td.mask.to(device_to_process)

    def reload(self, best_configuration):
        self.base.load_state_dict(best_configuration[0])

        # Base no trainable
        for p in self.base.conv1.parameters():
            p.requires_grad = False
        for p in self.base.conv2.parameters():
            p.requires_grad = False
        for p in self.base.conv3.parameters():
            p.requires_grad = False
        for p in self.base.conv4.parameters():
            p.requires_grad = False
        for p in self.base.bn1.parameters():
            p.requires_grad = False
        for p in self.base.bn2.parameters():
            p.requires_grad = False
        for p in self.base.bn3.parameters():
            p.requires_grad = False
        for p in self.base.bn4.parameters():
            p.requires_grad = False

        for i in range(self.get_ensemble_size()):
            self.branches[i].load_state_dict(best_configuration[i + 1])

            # Branch no trainable, but last layer
            for p in self.branches[i].conv1.parameters():
                p.requires_grad = False
            for p in self.branches[i].conv2.parameters():
                p.requires_grad = False
            for p in self.branches[i].conv3.parameters():
                p.requires_grad = False
            for p in self.branches[i].conv4.parameters():
                p.requires_grad = False
            for p in self.branches[i].bn1.parameters():
                p.requires_grad = False
            for p in self.branches[i].bn2.parameters():
                p.requires_grad = False
            for p in self.branches[i].bn3.parameters():
                p.requires_grad = False
            for p in self.branches[i].bn4.parameters():
                p.requires_grad = False
            for p in self.branches[i].fc.parameters():
                p.requires_grad = False
