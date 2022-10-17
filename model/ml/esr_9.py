#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of ESR-9 (Siqueira et al., 2020) trained on AffectNet (Mollahosseini et al., 2017) for emotion
and affect perception.


Reference:
    Siqueira, H., Magg, S. and Wermter, S., 2020. Efficient Facial Feature Learning with Wide Ensemble-based
    Convolutional Neural Networks. Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence
    (AAAI-20), pages 1–1, New York, USA.

    Mollahosseini, A., Hasani, B. and Mahoor, M.H., 2017. AffectNet: A database for facial expression, valence,
    and arousal computing in the wild. IEEE Transactions on Affective Computing, 10(1), pp.18-31.
"""

__author__ = "Henrique Siqueira"
__email__ = "siqueira.hc@outlook.com"
__license__ = "MIT license"
__version__ = "1.0"

# Standard libraries
from os import path

# External libraries
import torch.nn.functional as F
import torch.nn as nn
import torch

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR
from wrapyfi.config.manager import ConfigManager


class Base(nn.Module):
    """
        The base of the network (Ensembles with Shared Representations, ESRs) is responsible for learning low- and
        mid-level representations from the input data that are shared with an ensemble of convolutional branches
        on top of the architecture.

        In our paper (Siqueira et al., 2020), it is called shared layers or shared representations.
    """

    def __init__(self):
        super(Base, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)

        # Batch-normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        # Max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Convolutional, batch-normalization and pooling layers for representation learning
        x_shared_representations = F.relu(self.bn1(self.conv1(x)))
        x_shared_representations = self.pool(F.relu(self.bn2(self.conv2(x_shared_representations))))
        x_shared_representations = F.relu(self.bn3(self.conv3(x_shared_representations)))
        x_shared_representations = self.pool(F.relu(self.bn4(self.conv4(x_shared_representations))))

        return x_shared_representations


class ConvolutionalBranch(nn.Module):
    """
        Convolutional branches that compose the ensemble in ESRs. Each branch was trained on a sub-training
        set from the AffectNet dataset to learn complementary representations from the data (Siqueira et al., 2020).

        Note that, the second last layer provides eight discrete emotion labels whereas the last layer provides
        continuous values of arousal and valence levels.
    """

    def __init__(self):
        super(ConvolutionalBranch, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(128, 128, 3, 1)
        self.conv2 = nn.Conv2d(128, 256, 3, 1)
        self.conv3 = nn.Conv2d(256, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)

        # Batch-normalization layers
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # Second last, fully-connected layer related to discrete emotion labels
        self.fc = nn.Linear(512, 8)

        # Last, fully-connected layer related to continuous affect levels (arousal and valence)
        self.fc_dimensional = nn.Linear(8, 2)

        # Pooling layers
        # Max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_shared_representations):
        # Convolutional, batch-normalization and pooling layers
        x_conv_branch = F.relu(self.bn1(self.conv1(x_shared_representations)))
        x_conv_branch = self.pool(F.relu(self.bn2(self.conv2(x_conv_branch))))
        x_conv_branch = F.relu(self.bn3(self.conv3(x_conv_branch)))
        x_conv_branch = self.global_pool(F.relu(self.bn4(self.conv4(x_conv_branch))))
        x_conv_branch = x_conv_branch.view(-1, 512)

        # Fully connected layer for emotion perception
        discrete_emotion = self.fc(x_conv_branch)

        # Application of the ReLU function to neurons related to discrete emotion labels
        x_conv_branch = F.relu(discrete_emotion)

        # Fully connected layer for affect perception
        continuous_affect = self.fc_dimensional(x_conv_branch)

        # Returns activations of the discrete emotion output layer and arousal and valence levels
        return discrete_emotion, continuous_affect

    def forward_to_last_conv_layer(self, x_shared_representations):
        """
        Propagates activations to the last convolutional layer of the architecture.
        This method is used to generate saliency maps with the Grad-CAM algorithm (Selvaraju et al., 2017).

        Reference:
            Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D., 2017.
            Grad-cam: Visual explanations from deep networks via gradient-based localization.
            In Proceedings of the IEEE international conference on computer vision (pp. 618-626).

        :param x_shared_representations: (ndarray) feature maps from shared layers
        :return: feature maps of the last convolutional layer
        """

        # Convolutional, batch-normalization and pooling layers
        x_to_last_conv_layer = F.relu(self.bn1(self.conv1(x_shared_representations)))
        x_to_last_conv_layer = self.pool(F.relu(self.bn2(self.conv2(x_to_last_conv_layer))))
        x_to_last_conv_layer = F.relu(self.bn3(self.conv3(x_to_last_conv_layer)))
        x_to_last_conv_layer = F.relu(self.bn4(self.conv4(x_to_last_conv_layer)))

        # Feature maps of the last convolutional layer
        return x_to_last_conv_layer

    def forward_from_last_conv_layer_to_output_layer(self, x_from_last_conv_layer):
        """
        Propagates activations to the second last, fully-connected layer (here referred as output layer).
        This layer represents emotion labels.

        :param x_from_last_conv_layer: (ndarray) feature maps from the last convolutional layer of this branch.
        :return: (ndarray) activations of the last second, fully-connected layer of the network
        """

        # Global average polling and reshape
        x_to_output_layer = self.global_pool(x_from_last_conv_layer)
        x_to_output_layer = x_to_output_layer.view(-1, 512)

        # Output layer: emotion labels
        x_to_output_layer = self.fc(x_to_output_layer)

        # Returns activations of the discrete emotion output layer
        return x_to_output_layer


class ESR(MiddlewareCommunicator, nn.Module):
    """
    ESR is the unified ensemble architecture composed of two building blocks the Base and ConvolutionalBranch
    classes as described below by Siqueira et al. (2020):

    'An ESR consists of two building blocks. (1) The base (class Base) of the network is an array of convolutional
    layers for low- and middle-level feature learning. (2) These informative features are then shared with
    independent convolutional branches (class ConvolutionalBranch) that constitute the ensemble.'
    """

    # Default values
    # Input size
    INPUT_IMAGE_SIZE = (96, 96)
    # Values for pre-processing input data
    INPUT_IMAGE_NORMALIZATION_MEAN = [0.0, 0.0, 0.0]
    INPUT_IMAGE_NORMALIZATION_STD = [1.0, 1.0, 1.0]
    # Path to saved network
    PATH_TO_SAVED_NETWORK = "./model/ml/trained_models/esr_9"
    FILE_NAME_BASE_NETWORK = "Net-Base-Shared_Representations.pt"
    FILE_NAME_CONV_BRANCH = "Net-Branch_{}.pt"

    def __init__(self, device):
        """
        Loads ESR-9.

        :param device: Device to load ESR-9: GPU or CPU.
        """
        MiddlewareCommunicator.__init__(self)
        nn.Module.__init__(self)

        # Base of ESR-9 as described in the docstring (see mark 1)
        self.device = device
        self.base = None
        self.create_base(device)

        # Load 9 convolutional branches that composes ESR-9 as described in the docstring (see mark 2)
        self.convolutional_branches = {}
        for branch_idx in range(1, len(self) + 1):
            getattr(self, f"create_branch_{branch_idx}")(device)

        self.to(device)

        # Evaluation mode on
        self.eval()

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/created_base",
                                     should_wait=True)
    def create_base(self, device):
        """
        Creates the base of the network.
        """
        self.base = Base()
        self.base.load_state_dict(torch.load(path.join(ESR.PATH_TO_SAVED_NETWORK,
                                                       ESR.FILE_NAME_BASE_NETWORK),
                                             map_location=device))
        self.base.to(self.device)
        return True,

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/forward_base",
                                     carrier="", should_wait=True,
                                     listener_kwargs=dict(load_torch_device='$device'))
    def forward_base(self, x, device=None):
        return self.base(x),

    def _create_branch(self, device, branch_idx):
        """
        Creates convolutional branches that composes ESR-9.

        :param device: Device to load ESR-9: GPU or CPU.
        :return: 1 of 9 convolutional branches that composes ESR-9
        """

        self.convolutional_branches[branch_idx] = ConvolutionalBranch()
        self.convolutional_branches[branch_idx].load_state_dict(torch.load(path.join(ESR.PATH_TO_SAVED_NETWORK,
                                                                            ESR.FILE_NAME_CONV_BRANCH.format(branch_idx)),
                                                                  map_location=device))
        self.convolutional_branches[branch_idx].to(device)
        return branch_idx

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/created_branch_1",
                                     should_wait=True)
    def create_branch_1(self, device):
        return self._create_branch(device, 1),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/forward_branch_1",
                                     should_wait=True, listener_kwargs=dict(load_torch_device='$device'))
    def forward_branch_1(self, x_shared_representations, device=None):
        return self.convolutional_branches[1](x_shared_representations),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/created_branch_2",
                                     should_wait=True)
    def create_branch_2(self, device):
        return self._create_branch(device, 2),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/forward_branch_2",
                                     should_wait=True, listener_kwargs=dict(load_torch_device='$device'))
    def forward_branch_2(self, x_shared_representations, device=None):
        return self.convolutional_branches[2](x_shared_representations),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/created_branch_3",
                                     should_wait=True)
    def create_branch_3(self, device):
        return self._create_branch(device, 3),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/forward_branch_3",
                                     should_wait=True, listener_kwargs=dict(load_torch_device='$device'))
    def forward_branch_3(self, x_shared_representations, device=None):
        return self.convolutional_branches[3](x_shared_representations),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/created_branch_4",
                                     should_wait=True)
    def create_branch_4(self, device):
        return self._create_branch(device, 4),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/forward_branch_4",
                                     should_wait=True, listener_kwargs=dict(load_torch_device='$device'))
    def forward_branch_4(self, x_shared_representations, device=None):
        return self.convolutional_branches[4](x_shared_representations),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/created_branch_5",
                                     should_wait=True)
    def create_branch_5(self, device):
        return self._create_branch(device, 5),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/forward_branch_5",
                                     should_wait=True, listener_kwargs=dict(load_torch_device='$device'))
    def forward_branch_5(self, x_shared_representations, device=None):
        return self.convolutional_branches[5](x_shared_representations),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/created_branch_6",
                                     should_wait=True)
    def create_branch_6(self, device):
        return self._create_branch(device, 6),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/forward_branch_6",
                                     should_wait=True, listener_kwargs=dict(load_torch_device='$device'))
    def forward_branch_6(self, x_shared_representations, device=None):
        return self.convolutional_branches[6](x_shared_representations),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/created_branch_7",
                                     should_wait=True)
    def create_branch_7(self, device):
        return self._create_branch(device, 7),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/forward_branch_7",
                                     should_wait=True, listener_kwargs=dict(load_torch_device='$device'))
    def forward_branch_7(self, x_shared_representations, device=None):
        return self.convolutional_branches[7](x_shared_representations),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/created_branch_8",
                                     should_wait=True)
    def create_branch_8(self, device):
        return self._create_branch(device, 8),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/forward_branch_8",
                                     should_wait=True, listener_kwargs=dict(load_torch_device='$device'))
    def forward_branch_8(self, x_shared_representations, device=None):
        return self.convolutional_branches[8](x_shared_representations),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/created_branch_9",
                                     should_wait=True)
    def create_branch_9(self, device):
        return self._create_branch(device, 9),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "ESR", "/esr9/forward_branch_9",
                                     should_wait=True, listener_kwargs=dict(load_torch_device='$device'))
    def forward_branch_9(self, x_shared_representations, device=None):
        return self.convolutional_branches[9](x_shared_representations),

    def forward(self, x):
        """
        Forward method of ESR-9.

        :param x: (ndarray) Input data.
        :return: A list of emotions and affect values from each convolutional branch in the ensemble.
        """

        # List of emotions and affect values from the ensemble
        emotions = []
        affect_values = []

        # Get shared representations
        x_shared_representations, = self.forward_base(x, device=self.device)

        # Add to the lists of predictions outputs from each convolutional branch in the ensemble
        for branch_idx in range(1, len(self)+1):
            (output_emotion, output_affect), = getattr(self, f"forward_branch_{branch_idx}")\
                (x_shared_representations, device=self.device)  # branch(x_shared_representations)
            emotions.append(output_emotion)
            affect_values.append(output_affect)

        return emotions, affect_values

    def __len__(self):
        """
        ESR with nine branches trained on AffectNet (Siqueira et al., 2020).
        :return: (int) Size of the ensemble
        """
        return 9
