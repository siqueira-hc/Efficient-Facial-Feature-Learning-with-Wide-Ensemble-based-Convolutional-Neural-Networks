#!/usr/bin/env python
# -*- coding: utf-8 -*-

# External Libraries
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
import torch

# Standard Libraries
from os import path, makedirs
import copy

# Modules
from model.utils import udata, umath
from model.ml.esr_9 import ESR
from ensemble_network import Ensemble


def main():
    base_path_experiment = "./experiments/sample/"
    name_experiment = "ESR_9-sample"
    base_path_to_dataset = "./sample_dataset/"
    num_branches_trained_network = 9
    validation_interval = 2
    max_training_epoch = 2
    current_branch_on_training = 8

     # Make dir
    if not path.isdir(path.join(base_path_experiment, name_experiment)):
        makedirs(path.join(base_path_experiment, name_experiment))

    # Define transforms
    data_transforms = [transforms.ColorJitter(brightness=0.5, contrast=0.5),
                       transforms.RandomHorizontalFlip(p=0.5),
                       transforms.RandomAffine(degrees=30,
                                               translate=(.1, .1),
                                               scale=(1.0, 1.25),
                                               resample=Image.BILINEAR)]

    # Running device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Starting: {}".format(str(name_experiment)))
    print("Running on {}".format(device))

    # Load network trained on AffectNet
    net = Ensemble.load(device, num_branches_trained_network)

    # Send params to device
    net.to_device(device)

    # Set optimizer
    optimizer = optim.SGD([{"params": net.base.parameters(), "lr": 0.1, "momentum": 0.9},
                           {"params": net.branches[0].parameters(), "lr": 0.1, "momentum": 0.9}])
    for b in range(1, net.get_ensemble_size()):
        optimizer.add_param_group({"params": net.branches[b].parameters(), "lr": 0.02, "momentum": 0.9})

    # Define criterion
    criterion = nn.CrossEntropyLoss()

    # Load validation set
    # max_loaded_images_per_label=100000 loads the whole validation set
    val_data = udata.Sample(idx_set=1,
                            max_loaded_images_per_label=1000,
                            transforms=None,
                            base_path_to_sample=base_path_to_dataset)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8)

    # Fine-tune ESR-9
    for branch_on_training in range(num_branches_trained_network):
        # Load training data
        train_data = udata.Sample(idx_set=0,
                                  max_loaded_images_per_label=5000,
                                  transforms=transforms.Compose(data_transforms),
                                  base_path_to_sample=base_path_to_dataset)

        # Best network
        best_ensemble = net.to_state_dict()
        best_ensemble_acc = 0.0

        # Initialize scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75, last_epoch=-1)

        # History
        history_loss = []
        history_acc = [[] for _ in range(net.get_ensemble_size())]
        history_val_loss = [[] for _ in range(net.get_ensemble_size())]
        history_val_acc = [[] for _ in range(net.get_ensemble_size() + 1)]

        # Training branch
        for epoch in range(max_training_epoch):
            train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=8)

            running_loss = 0.0
            running_corrects = [0.0 for _ in range(net.get_ensemble_size())]
            running_updates = 0

            scheduler.step()

            for inputs, labels in train_loader:
                # Get the inputs
                # print(inputs, labels)
                inputs, labels = inputs.to(device), labels.to(device)

                # Set gradients to zero
                optimizer.zero_grad()

                # Forward
                outputs = net(inputs)
                confs_preds = [torch.max(o, 1) for o in outputs]

                # Compute loss
                loss = 0.0
                for i_4 in range(net.get_ensemble_size() - current_branch_on_training):
                    preds = confs_preds[i_4][1]
                    running_corrects[i_4] += torch.sum(preds == labels).cpu().numpy()
                    loss += criterion(outputs[i_4], labels)

                # Backward
                loss.backward()

                # Optimize
                optimizer.step()

                # Save loss
                running_loss += loss.item()
                running_updates += 1

            # Statistics
            print("[Branch {:d}, Epochs {:d}--{:d}] "
                  "Loss: {:.4f} Acc: {}".format(net.get_ensemble_size() - current_branch_on_training,
                                                epoch + 1,
                                                max_training_epoch,
                                                running_loss / running_updates,
                                                np.array(running_corrects) / len(train_data)))
            # # Validation
            # if ((epoch % validation_interval) == 0) or ((epoch + 1) == max_training_epoch):
            #     net.eval()

            #     val_loss, val_corrects = evaluate(net, val_loader, criterion, device, current_branch_on_training)

            #     print("\nValidation - [Branch {:d}, Epochs {:d}--{:d}] Loss: {:.4f} Acc: {}\n\n".format(
            #         net.get_ensemble_size() - current_branch_on_training,
            #         epoch + 1,
            #         max_training_epoch,
            #         val_loss[-1],
            #         np.array(val_corrects) / len(val_data)))

            #     # Add to history training and validation statistics
            #     history_loss.append(running_loss / running_updates)

            #     for i_4 in range(net.get_ensemble_size()):
            #         history_acc[i_4].append(running_corrects[i_4] / len(train_data))

            #     for b in range(net.get_ensemble_size()):
            #         history_val_loss[b].append(val_loss[b])
            #         history_val_acc[b].append(float(val_corrects[b]) / len(val_data))

            #     # Add ensemble accuracy to history
            #     history_val_acc[-1].append(float(val_corrects[-1]) / len(val_data))

            #     # Save best ensemble
            #     ensemble_acc = (float(val_corrects[-1]) / len(val_data))
            #     if ensemble_acc >= best_ensemble_acc:
            #         best_ensemble_acc = ensemble_acc
            #         best_ensemble = net.to_state_dict()

            #         # Save network
            #         Ensemble.save(best_ensemble,
            #                       path.join(base_path_experiment, name_experiment, "Saved Networks"),
            #                       current_branch_on_training)

            #     # Save graphs
            #     plot(history_loss,
            #          history_acc,
            #          history_val_loss,
            #          history_val_acc,
            #          net.get_ensemble_size() - current_branch_on_training,
            #          path.join(base_path_experiment, name_experiment))

            #     net.train()

        # Change branch on training
        if current_branch_on_training > 0:
            # Decrease max training epoch
            max_training_epoch = 2

            # Reload best configuration
            net.reload(best_ensemble)

            # Set optimizer
            optimizer = optim.SGD([{"params": net.base.parameters(), "lr": 0.02, "momentum": 0.9},
                                   {"params": net.branches[
                                       net.get_ensemble_size() - current_branch_on_training].parameters(),
                                    "lr": 0.1,
                                    "momentum": 0.9
                                    }])
            # Trained branches
            for b in range(net.get_ensemble_size()):
                if b != (net.get_ensemble_size() - current_branch_on_training):
                    optimizer.add_param_group({"params": net.branches[b].parameters(), "lr": 0.02, "momentum": 0.9})

            # Change branch on training
            current_branch_on_training -= 1

        # Finish training after fine-tuning all branches
        else:
            break


if __name__ == '__main__':
    main()
