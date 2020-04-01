#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements methods to handle datasets.
"""

__author__ = "Henrique Siqueira"
__email__ = "siqueira.hc@outlook.com"
__license__ = "MIT license"
__version__ = "0.3"

# External Libraries
from torch.utils.data import Dataset


# TODO: On development

class AffectNetCategorical(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None

    def _load(self):
        pass

    @staticmethod
    def get_class(idx):
        classes = {
            0: "Neutral",
            1: "Happy",
            2: "Sad",
            3: "Surprise",
            4: "Fear",
            5: "Disgust",
            6: "Anger",
            7: "Contempt"}

        return classes[idx]
