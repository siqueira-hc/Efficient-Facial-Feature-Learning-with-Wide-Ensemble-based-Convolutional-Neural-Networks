#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script of the facial expression recognition framework.

It has three main features:

Image: recognizes facial expressions in images.

Video: recognizes facial expressions in videos in a frame-based approach.

Webcam: connects to a webcam and recognizes facial expressions of the closest face detected
by a face detection algorithm.
"""

__author__ = "Henrique Siqueira"
__email__ = "siqueira.hc@outlook.com"
__license__ = "MIT license"
__version__ = "1.0"

# Standard Libraries
import argparse
from argparse import RawTextHelpFormatter
import time
import os

# Modules
from controller import cvalidation, cvision
from model.utils import uimage, ufile
from model.screen.fer_demo import FERDemo


def webcam(camera_id, display, gradcam, output_csv_file, screen_size, device, frames, branch, no_plot, face_detection):
    """
    Receives images from a camera and recognizes
    facial expressions of the closets face in a frame-based approach.
    """

    fer_demo = None
    write_to_file = not (output_csv_file is None)
    starting_time = time.time()

    if not uimage.initialize_video_capture(camera_id):
        print("this is where I failed!")
        raise RuntimeError("Error on initializing video capture." +
                           "\nCheck whether a webcam is working or not." +
                           "In linux, you can use Cheese for testing.")

    uimage.set_fps(frames)

    # Initialize screen
    if display:
        fer_demo = FERDemo(screen_size=screen_size,
                           display_individual_classification=branch,
                           display_graph_ensemble=(not no_plot))
    else:
        print("Press 'Ctrl + C' to quit.")

    try:
        if write_to_file:
            ufile.create_file(output_csv_file, str(time.time()))

        # Loop to process each frame from a VideoCapture object.
        while uimage.is_video_capture_open() and ((not display) or (display and fer_demo.is_running())):
            # Get a frame
            img, _ = uimage.get_frame()

            fer = None if (img is None) else cvision.recognize_facial_expression(img, device, face_detection, gradcam)

            # Display blank screen if no face is detected, otherwise,
            # display detected faces and perceived facial expression labels
            if display:
                fer_demo.update(fer)
                fer_demo.show()

            if write_to_file:
                ufile.write_to_file(fer, time.time() - starting_time)

    except Exception as e:
        print("Error raised during video mode.")
        raise e
    except KeyboardInterrupt as qe:
        print("Keyboard interrupt event raised.")
    finally:
        uimage.release_video_capture()

        if display:
            fer_demo.quit()

        if write_to_file:
            ufile.close_file()


def image(input_image_path, display, gradcam, output_csv_file, screen_size, device, branch, face_detection):
    """
    Receives the full path to a image file and recognizes
    facial expressions of the closets face in a frame-based approach.
    """

    write_to_file = not (output_csv_file is None)
    img = uimage.read(input_image_path)

    # Call FER method
    fer = cvision.recognize_facial_expression(img, device, face_detection, gradcam)

    if write_to_file:
        ufile.create_file(output_csv_file, input_image_path)
        ufile.write_to_file(fer, 0.0)
        ufile.close_file()
        fer_demo = FERDemo(screen_size=screen_size,
                           display_individual_classification=branch,
                           display_graph_ensemble=False)
        fer_demo.update(fer)
        # while fer_demo.is_running():
        fer_demo.save()
        fer_demo.quit()

    if display:
        fer_demo = FERDemo(screen_size=screen_size,
                           display_individual_classification=branch,
                           display_graph_ensemble=False)
        fer_demo.update(fer)
        while fer_demo.is_running():
            fer_demo.show()
        fer_demo.quit()


def video(input_video_path, display, gradcam, output_csv_file, screen_size,
          device, frames, branch, no_plot, face_detection, emotion_cat=""):
    """
    Receives the full path to a video file and recognizes
    facial expressions of the closets face in a frame-based approach.
    """

    fer_demo = None
    write_to_file = not (output_csv_file is None)

    if not uimage.initialize_video_capture(input_video_path):
        raise RuntimeError("Error on initializing video capture." +
                           "\nCheck whether working versions of ffmpeg or gstreamer is installed." +
                           "\nSupported file format: MPEG-4 (*.mp4).")

    uimage.set_fps(frames)

    # Initialize screen
    if display:
        fer_demo = FERDemo(screen_size=screen_size,
                           display_individual_classification=branch,
                           display_graph_ensemble=(not no_plot))

    try:
        if write_to_file:
            ufile.create_file(output_csv_file, input_video_path, emotion_cat)

        # Loop to process each frame from a VideoCapture object.
        while uimage.is_video_capture_open() and ((not display) or (display and fer_demo.is_running())):
            # Get a frame
            img, timestamp = uimage.get_frame()

            # Video has been processed
            if img is None:
                break
            else:  # Process frame
                fer = None if (img is None) else cvision.recognize_facial_expression(img,
                                                                                     device,
                                                                                     face_detection,
                                                                                     gradcam)

                # Display blank screen if no face is detected, otherwise,
                # display detected faces and perceived facial expression labels
                if display:
                    fer_demo.update(fer)
                    fer_demo.show()

                if write_to_file:
                    ufile.write_to_file(fer, timestamp)

    except Exception as e:
        print("Error raised during video mode.")
        raise e
    finally:
        uimage.release_video_capture()

        if display:
            fer_demo.quit()

        if write_to_file:
            ufile.close_file()


def eval_video(input_video_path, display, gradcam, output_dir, screen_size,
               device, frames, branch, no_plot, face_detection):
    """
    Receives the full path to a video evaluation folder and recognizes
    facial expressions of the closest face in a frame-based approach.
    """
    write_to_file = not (output_dir is None)

    # for each folder in input_video_path, and for each file in that folder, 
    # process the video file and store the output csv in a relevant location.
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    for emotion_dir in os.listdir(input_video_path):
        if os.path.isdir(os.path.join(input_video_path, emotion_dir)):
            for filename in os.listdir(os.path.join(input_video_path, emotion_dir)):
                eval_path = os.path.join(input_video_path, emotion_dir, filename)
                print("Input path is: " + eval_path)
                emotion_cat = emotion_dir[:2]
                video(eval_path, display, gradcam, output_dir, screen_size, device, frames, branch,
                    no_plot, face_detection, emotion_cat)
    return None


def main():
    # Parser
    parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)
    parser.add_argument("mode", help="select a method among 'image', 'video' or 'webcam' to run ESR-9.",
                        type=str, choices=["image", "video", "webcam", "eval_image", "eval_video"])
    parser.add_argument("-d", "--display", help="display the output of ESR-9.",
                        action="store_true")
    parser.add_argument("-g", "--gradcam", help="run grad-CAM and displays the salience maps.",
                        action="store_true")
    parser.add_argument("-i", "--input", help="define the full path to an image or video.",
                        type=str)
    parser.add_argument("-o", "--output",
                        help="create and write ESR-9's outputs to a CSV file. The file is saved in a folder defined "
                             "by this argument (ex. '-o ./' saves the file with the same name as the input file "
                             "in the working directory).",
                        type=str)
    parser.add_argument("-s", "--size",
                        help="define the size of the window: \n1 - 1920 x 1080;\n2 - 1440 x 900;\n3 - 1024 x 768.",
                        type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("-c", "--cuda", help="run on GPU.",
                        action="store_true")
    parser.add_argument("-w", "--webcam_id",
                        help="define the webcam by 'id' to capture images in the webcam mode." +
                             "If none is selected, the default camera by the OS is used.",
                        type=int, default=-1)
    parser.add_argument("-f", "--frames", help="define frames of videos and webcam captures.",
                        type=int, default=5)
    parser.add_argument("-b", "--branch", help="show individual branch's classification if set true, otherwise," +
                                               "show final ensemble's classification.",
                        action="store_true", default=False)
    parser.add_argument("-np", "--no_plot", help="do not display activation and (un)pleasant graph",
                        action="store_true", default=False)

    parser.add_argument("-fd", "--face_detection",
                        help="define the face detection algorithm:" +
                             "\n1 - Optimized Dlib." +
                             "\n2 - Standard Dlib (King, 2009)." +
                             "\n3 - Haar Cascade Classifiers (Viola and Jones, 2004)." +
                             "\n[Warning] Dlib is slower but accurate, whereas haar cascade is faster "
                             "but less accurate",
                        type=int, choices=[1, 2, 3], default=1)

    args = parser.parse_args()

    # Calls to main methods
    if args.mode == "image":
        try:
            cvalidation.validate_image_video_mode_arguments(args)
            image(args.input, args.display, args.gradcam, args.output,
                  args.size, args.cuda, args.branch, args.face_detection)
        except RuntimeError as e:
            print(e)
    elif args.mode == "video":
        try:
            cvalidation.validate_image_video_mode_arguments(args)
            video(args.input, args.display, args.gradcam, args.output,
                  args.size, args.cuda, args.frames, args.branch, args.no_plot, args.face_detection)
        except RuntimeError as e:
            print(e)
    elif args.mode == "webcam":
        try:
            cvalidation.validate_webcam_mode_arguments(args)
            webcam(args.webcam_id, args.display, args.gradcam, args.output,
                   args.size, args.cuda, args.frames, args.branch, args.no_plot, args.face_detection)
        except RuntimeError as e:
            print(e)
    elif args.mode == "eval_video":
        try:
            cvalidation.validate_image_video_mode_arguments(args)
            eval_video(args.input, args.display, args.gradcam, args.output,
                       args.size, args.cuda, args.frames, args.branch, args.no_plot, args.face_detection)
        except RuntimeError as e:
            print(e)


if __name__ == "__main__":
    print("Processing...")
    main()
    print("Process has finished!")
