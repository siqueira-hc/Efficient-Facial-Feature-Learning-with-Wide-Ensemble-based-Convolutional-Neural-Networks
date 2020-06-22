#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script with various eval functions that can be called for 
performing different types of evaluations. 
"""

# Standard libraries
import argparse
from argparse import RawTextHelpFormatter
import os
import operator
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import classification_report

# Modules
from model.utils import uimage, ufile
from controller import cvision


def eval_frame(predictions_dir, emotion_dict):
    """
    Evaluate the predictions of the model on an evaluation dataset, frame by frame. 
    """
    emotion_to_num = {"neutral": 0, "happy": 1, "sad": 2, "surprise": 3,
                     "fear": 4, "disgust": 5,  "anger": 6, "contempt": 7}
    labels, preds = [], []
    for filename in os.listdir(predictions_dir):
        if filename.endswith(".csv"):
            # Get the ground truth label from the filename for now
            emotion_label = emotion_dict[filename[:2]]
            emotion_num = emotion_to_num[emotion_label]

            # Get the predicted emotions, frame by frame
            full_file_path = os.path.join(predictions_dir, filename)
            print(full_file_path)
            preds_df = pd.read_csv(full_file_path)
            pred_emotions = preds_df['Ensemble_Emotion'].tolist()
            # pred_emotions = pred_emotions[:-1]
            pred_nums = []
            for emotion in pred_emotions:
                if emotion.lower() == "none":
                    continue
                pred_nums.append(emotion_to_num[emotion.lower()])
            # pred_nums = [emotion_to_num[emotion.lower()] for emotion in pred_emotions]

            # Extend the labels and preds lists
            num_frames = len(pred_nums)
            labels.extend([emotion_num] * num_frames)
            preds.extend(pred_nums)
    print(classification_report(labels, preds))


def eval_forward_frame(data_dir, emotion_dict, img_type='npy'):
    """
    Given a data directory (of images), generate the predictions for each image in the directory 
    (and its subdirectories), and use these predictions for evaluation. 
    Store the predictions in a prediction file and report the evaluation performance. 
    """
    emotion_to_num = {"neutral": 0, "happy": 1, "sad": 2, "surprise": 3,
                     "fear": 4, "disgust": 5,  "anger": 6, "contempt": 7}
    labels, preds = [], []

    for emotion_dir in os.listdir(data_dir)[:1]:
        for img_npy in os.listdir(os.path.join(data_dir, emotion_dir)):
            # Get the ground truth emotion label 
            emotion_label = emotion_dict[emotion_dir[:2]]
            emotion_num = emotion_to_num[emotion_label]

            # Read the image and predict facial expression.
            if img_type == 'npy':
                img = np.load(os.path.join(data_dir, emotion_dir, img_npy))
            elif img_type == 'jpg' or img_type == 'png':
                img = cv2.imread(os.path.join(data_dir, emotion_dir, img_npy))
            # The second argument is False since we are currently testing on CPU
            emotion, affect, emotion_idx = cvision.recognize_face_image_expression(img, False)
            # print(emotion, emotion_idx)
            ensemble_prediction = emotion_idx[-1]
            labels.append(emotion_num)
            preds.append(ensemble_prediction)
    print(classification_report(labels, preds))


def eval_video(predictions_dir, emotion_dict):
    """
    Evaluate the predictions of the model taking a video as a whole. 
    For now, we consider majority emotion through the video as the prediction. 
    """
    emotion_to_num = {"neutral": 0, "happy": 1, "sad": 2, "surprise": 3,
                     "fear": 4, "disgust": 5,  "anger": 6, "contempt": 7}
    labels, preds = [], []
    for filename in os.listdir(predictions_dir):
        if filename.endswith(".csv"):
            # Get the ground truth label from the filename for now
            emotion_label = emotion_dict[filename[:2]]
            emotion_num = emotion_to_num[emotion_label]

            # Get the predicted emotions, frame by frame
            full_file_path = os.path.join(predictions_dir, filename)
            print(full_file_path)
            preds_df = pd.read_csv(full_file_path)
            pred_emotions = preds_df['Ensemble_Emotion'].tolist()
            pred_nums = {}
            for emotion in pred_emotions:
                if emotion.lower() == "none":
                    continue
                pred_emotion_num = emotion_to_num[emotion.lower()]
                if pred_emotion_num not in pred_nums:
                    pred_nums[pred_emotion_num] = 1
                else:
                    pred_nums[pred_emotion_num] += 1
            if len(pred_nums.keys()) == 0:
                continue

            # Pick the maximum frequency emotion as the prediction for this video. 
            prediction = max(pred_nums.items(), key=operator.itemgetter(1))[0]
            preds.append(prediction)
            labels.append(emotion_num)
    print(classification_report(labels, preds))


def main():
    # Parser
    parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)
    parser.add_argument("-p", "--predictions", help="directory where all the predictions/data points are stored.", 
                        type=str)
    parser.add_argument("-em", "--eval_mode", 
                        help="Evaluation mode: "
                            "(1) frame by frame evaluation, with the label being the same for each frame. "
                            "(2) evaluation of the whole video prediction as a whole through some aggregation.",
                        type=str, choices=["frame", "video"])
    parser.add_argument("-f", "--forward", help="If True, the model needs a forward pass over the frames first", 
                        type=str)
    args = parser.parse_args()

    emotion_dict = {"ha": "happy", "sa": "sad", "ne": "neutral", "su": "surprise",
                    "an": "anger", "fe": "fear", "di": "disgust", "co": "contempt"}
    if args.forward.lower().strip() == "true":
        eval_forward_frame(args.predictions, emotion_dict)
        return 
    if args.eval_mode == "frame":
        eval_frame(args.predictions, emotion_dict)
    elif args.eval_mode == "video":
        eval_video(args.predictions, emotion_dict)

if __name__ == "__main__":
    main()
