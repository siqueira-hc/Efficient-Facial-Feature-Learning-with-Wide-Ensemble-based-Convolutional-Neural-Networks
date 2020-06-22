#!/usr/bin/env python
# -*- coding: utf-8 -*-

# External Libraries
import torch.nn.functional as F
import torch.nn as nn
import torch

# Standard Libraries
from os import path, makedirs
import copy

# Modules
from model.ml.esr_9 import ESR


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

    def forward(self, x_base_to_process):
        x_base = F.relu(self.bn1(self.conv1(x_base_to_process)))
        x_base = self.pool(F.relu(self.bn2(self.conv2(x_base))))
        x_base = F.relu(self.bn3(self.conv3(x_base)))
        x_base = self.pool(F.relu(self.bn4(self.conv4(x_base))))

        return x_base


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

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_branch_to_process):
        x_branch = F.relu(self.bn1(self.conv1(x_branch_to_process)))
        x_branch = self.pool(F.relu(self.bn2(self.conv2(x_branch))))
        x_branch = F.relu(self.bn3(self.conv3(x_branch)))
        x_branch = self.global_pool(F.relu(self.bn4(self.conv4(x_branch))))
        x_branch = x_branch.view(-1, 512)
        x_branch = self.fc(x_branch)

        return x_branch


class Ensemble(nn.Module):
    def __init__(self):
        super(Ensemble, self).__init__()

        self.base = Base()
        self.branches = []

    def get_ensemble_size(self):
        return len(self.branches)

    def add_branch(self):
        self.branches.append(Branch())

    def forward(self, x):
        x_ensemble = self.base(x)

        y = []
        for branch in self.branches:
            y.append(branch(x_ensemble))

        return y

    @staticmethod
    def save(state_dicts, base_path_to_save_model, current_branch_save):
        if not path.isdir(path.join(base_path_to_save_model, str(len(state_dicts) - 1 - current_branch_save))):
            makedirs(path.join(base_path_to_save_model, str(len(state_dicts) - 1 - current_branch_save)))

        torch.save(state_dicts[0],
                   path.join(base_path_to_save_model,
                             str(len(state_dicts) - 1 - current_branch_save),
                             "Net-Base-Shared_Representations.pt"))

        for i in range(1, len(state_dicts)):
            torch.save(state_dicts[i],
                       path.join(base_path_to_save_model,
                                 str(len(state_dicts) - 1 - current_branch_save),
                                 "Net-Branch_{}.pt".format(i)))

        print("Network has been "
              "saved at: {}".format(path.join(base_path_to_save_model,
                                              str(len(state_dicts) - 1 - current_branch_save))))

    @staticmethod
    def load(device_to_load, ensemble_size, load_path=None):
        # Load ESR-9
        if not path:       
            esr_9 = ESR(device_to_load, load_path)
        loaded_model = Ensemble()
        loaded_model.branches = []

        # Load the base of the network
        loaded_model.base = esr_9.base

        # Load branches
        for i in range(ensemble_size):
            loaded_model_branch = Branch()
            loaded_model_branch.conv1 = esr_9.convolutional_branches[i].conv1
            loaded_model_branch.conv2 = esr_9.convolutional_branches[i].conv2
            loaded_model_branch.conv3 = esr_9.convolutional_branches[i].conv3
            loaded_model_branch.conv4 = esr_9.convolutional_branches[i].conv4
            loaded_model_branch.bn1 = esr_9.convolutional_branches[i].bn1
            loaded_model_branch.bn2 = esr_9.convolutional_branches[i].bn2
            loaded_model_branch.bn3 = esr_9.convolutional_branches[i].bn3
            loaded_model_branch.bn4 = esr_9.convolutional_branches[i].bn4
            loaded_model_branch.fc = esr_9.convolutional_branches[i].fc
            loaded_model.branches.append(loaded_model_branch)

        return loaded_model

    def to_state_dict(self):
        state_dicts = [copy.deepcopy(self.base.state_dict())]

        for b in self.branches:
            state_dicts.append(copy.deepcopy(b.state_dict()))

        return state_dicts

    def to_device(self, device_to_process="cpu"):
        self.to(device_to_process)
        self.base.to(device_to_process)

        for b_td in self.branches:
            b_td.to(device_to_process)

    def reload(self, best_configuration):
        self.base.load_state_dict(best_configuration[0])

        for i in range(self.get_ensemble_size()):
            self.branches[i].load_state_dict(best_configuration[i + 1])