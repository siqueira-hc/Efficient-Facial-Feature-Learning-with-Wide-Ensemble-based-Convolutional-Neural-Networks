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
from sklearn.metrics import classification_report

# Modules


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
    parser.add_argument("-p", "--predictions", help="directory where all the predictions are stored.", type=str)
    parser.add_argument("-em", "--eval_mode", 
                        help="Evaluation mode: "
                            "(1) frame by frame evaluation, with the label being the same for each frame. "
                            "(2) evaluation of the whole video prediction as a whole through some aggregation.",
                        type=str, choices=["frame", "video"])
    args = parser.parse_args()

    emotion_dict = {"ha": "happy", "sa": "sad", "ne": "neutral", "su": "surprise",
                    "an": "anger", "fe": "fear", "di": "disgust", "co": "contempt"}
    if args.eval_mode == "frame":
        eval_frame(args.predictions, emotion_dict)
    elif args.eval_mode == "video":
        eval_video(args.predictions, emotion_dict)


if __name__ == "__main__":
    main()
