#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements computer vision algorithms.
"""

__author__ = "Henrique Siqueira"
__email__ = "siqueira.hc@outlook.com"
__license__ = "MIT license"
__version__ = "0.1"

# External Libraries
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2

# Modules
from model.esr.fer import FER
from model.utils import uimage, udata
from model.esr.esr_9 import Ensemble


# Haar-cascade Fast, Slow, Very-Slow

# Default values
_SCALE_FACTORS = (10.0, 1.3)
_INITIAL_NEIGHBORS = (205, 35)
_DECREMENT_NEIGHBORS = (-50, -10)
_MIN_NEIGHBORS = (10, 5)
_MIN_SIZE = (60, 60)
_MAX_SIZE = (600, 600)

# TODO: Temporary
"""
_SCALE_FACTOR = 1.05
_MIN_NEIGHBORS = 30
_MIN_SIZE = (60, 60)
_MAX_SIZE = (1024, 1024)
"""

# Private variables
_FACE_DETECTOR_HAAR_CASCADE = None
_ESR_9 = None


# Public methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def detect_face(image):
    """
    TODO: Docstring

    :param image:
    :return:
    """
    return _haar_cascade(image)


def recognize_facial_expression(image, on_gpu):
    """
    This method tries to detect a face in the input image.
    If more than one face is detected, the closets is used.
    Afterwards, the detected face is fed to ESR-9 for facial expression recognition.
    The face detection phase relies on third-party methods and ESR-9 does not verify
    if a face is used as input or not (false-positive cases).

    :param on_gpu:
    :param image: (ndarray) input image.
    :return: An FER object with the components necessary for display.
    """

    to_return_fer = None

    # Detect face
    face_coordinates = detect_face(image)

    if face_coordinates is None:
        to_return_fer = FER(image)
    else:
        face = image[face_coordinates[0][1]:face_coordinates[1][1], face_coordinates[0][0]:face_coordinates[1][0], :]

        # Get device
        device = torch.device("cuda" if on_gpu else "cpu")

        # Pre_process detected face
        input_face = _pre_process_input_image(face)
        input_face = input_face.to(device)

        # Recognize facial expression
        emotion, affect = _predict(input_face, device)

        # Initialize GUI object
        to_return_fer = FER(image, face, face_coordinates, emotion, affect)

    return to_return_fer

# Public methods <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# Private methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def _haar_cascade(image, scale_factors=_SCALE_FACTORS, initial_neighbors=_INITIAL_NEIGHBORS,
                  min_size=_MIN_SIZE, max_size=_MAX_SIZE):
    """
    Face detection using the Haar Feature-based Cascade Classifiers (Viola and Jones, 2004).

    References:
    Viola, P. and Jones, M. J. (2004). Robust real-time face detection. International journal of computer vision, 57(2), 137-154.

    :param image: (ndarray) input image.
    :param scale_factors:
    :param initial_neighbors:
    :param min_size:
    :param max_size:
    :return: (ndarray) If at least one face is detected, the method returns the coordinates of the closets face.
    """
    global _FACE_DETECTOR_HAAR_CASCADE

    # Verify if haar cascade is initialized
    if _FACE_DETECTOR_HAAR_CASCADE is None:
        _FACE_DETECTOR_HAAR_CASCADE = cv2.CascadeClassifier("./model/utils/templates/haar_cascade/frontal_face.xml")

    closest_face_area = 0
    face_coordinates = None

    greyscale_image = uimage.convert_bgr_to_grey(image)

    # TODO: Create an arg
    trials = len(scale_factors)

    for t in range(trials):
        for n in range(initial_neighbors[t], _MIN_NEIGHBORS[t] - 1, _DECREMENT_NEIGHBORS[t]):
            # TODO: Debugging
            # print("scale: " + str(scale_factors[t]) + "          n: " + str(n))

            faces = _FACE_DETECTOR_HAAR_CASCADE.detectMultiScale(greyscale_image, scale_factors[t], n, minSize=min_size, maxSize=max_size)

            # Look for the closest face and return its coordinates
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    (xi, yi, xf, yf) = (x, y, x + w, y + h)
                    face_area = (xf - xi) * (yf - yi)

                    if face_area > closest_face_area:
                        closest_face_area = face_area
                        face_coordinates = [(xi, yi), (xf, yf)]
                return face_coordinates

    # In case of no faces
    return None


def _pre_process_input_image(image):
    """
    TODO: Docstring

    :param image:
    :return:
    """

    image = uimage.resize(image, Ensemble.INPUT_IMAGE_SIZE)
    image = Image.fromarray(image)
    image = transforms.Normalize(mean=Ensemble.INPUT_IMAGE_NORMALIZATION_MEAN,
                                 std=Ensemble.INPUT_IMAGE_NORMALIZATION_STD)(transforms.ToTensor()(image)).unsqueeze(0)

    return image


def _predict(input_face, device):
    global _ESR_9

    if _ESR_9 is None:
        _ESR_9 = Ensemble.load(device)

    to_return_emotion = []
    to_return_affect = None

    # Recognize facial expression
    emotion, affect = _ESR_9(input_face)

    # Compute ensemble prediction for affect
    # Convert from Tensor to ndarray
    affect = np.array([a[0].cpu().detach().numpy() for a in affect])
    # Normalize arousal
    affect[:, 1] = np.clip((affect[:, 1] + 1)/2.0, 0, 1)
    # Compute mean arousal and valence as the ensemble prediction
    ensemble_affect = np.expand_dims(np.mean(affect, 0), axis=0)
    # Concatenate the ensemble prediction to the list of affect predictions
    to_return_affect = np.concatenate((affect, ensemble_affect), axis=0)

    # Compute ensemble prediction for emotion
    # Convert from Tensor to ndarray
    emotion = np.array([e[0].cpu().detach().numpy() for e in emotion])
    # Get number of classes
    num_classes = emotion.shape[1]
    # Compute votes and add label to the list of emotions
    emotion_votes = np.zeros(num_classes)
    for e in emotion:
        e_idx = np.argmax(e)
        to_return_emotion.append(udata.AffectNetCategorical.get_class(e_idx))
        emotion_votes[e_idx] += 1

    # Concatenate the ensemble prediction to the list of emotion predictions
    to_return_emotion.append(udata.AffectNetCategorical.get_class(np.argmax(emotion_votes)))

    return to_return_emotion, to_return_affect

# Private methods <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
