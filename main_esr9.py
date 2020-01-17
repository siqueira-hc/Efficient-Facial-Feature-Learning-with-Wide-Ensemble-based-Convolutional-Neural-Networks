#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO: Write docstring.
"""

__author__ = "Henrique Siqueira"
__email__ = "siqueira.hc@outlook.com"
__license__ = "MIT license"
__version__ = "0.1"

# Standard Libraries
import argparse

# Modules
from controller import cvalidation, cvision
from model.utils import uimage
from model.screen.fer_demo import FERDemo


def webcam(camera_id, display, gradcam, output_csv_file, screen_size, device, fps, branch, no_plot):
    """
    This method receives images from a camera and recognizes
    facial expressions of the closets face in a frame-based approach.

    TODO: Write docstring.
    :param no_plot:
    :param camera_id:
    :param display:
    :param gradcam:
    :param output_csv_file:
    :param screen_size:
    :param device:
    :param fps:
    :param branch:
    :return:
    """
    fer_demo = None

    if not uimage.initialize_video_capture(camera_id):
        raise RuntimeError("Error on initializing video capture." +
                           "\nCheck whether a webcam is working or not." +
                           "In linux, you can use Cheese for testing.")

    uimage.set_fps(fps)

    # Initialize screen
    if display:
        fer_demo = FERDemo(screen_size=screen_size, display_individual_classification=branch, display_graph_ensemble=(not no_plot))

    try:
        # Loop to process each frame from a VideoCapture object.
        while uimage.is_video_capture_open() and ((not display) or (display and fer_demo.is_running())):
            # Get a frame
            image = uimage.get_frame()

            fer = None if (image is None) else cvision.recognize_facial_expression(image, device)

            # Display blank screen if no face is detected, otherwise,
            # display detected faces and perceived facial expression labels
            if display:
                fer_demo.update(fer)
                fer_demo.show()

            # TODO: Implement
            if output_csv_file:
                pass

    except Exception as e:
        print("Error raised during video mode.")
        raise e
    finally:
        uimage.release_video_capture()
        fer_demo.quit()


def image(input_image_path, display, gradcam, output_csv_file, screen_size, device, branch):
    """
    This method receives the full path to a image file and recognizes
    facial expressions of the closets face in a frame-based approach.

    TODO: Write docstring.

    :param input_image_path:
    :param display:
    :param gradcam:
    :param output_csv_file:
    :param screen_size:
    :param device:
    :param branch:
    :return:
    """

    image = uimage.read(input_image_path)

    # Call FER method
    fer = cvision.recognize_facial_expression(image, device)

    # TODO: Implement
    if output_csv_file:
        pass

    if display:
        fer_demo = FERDemo(screen_size=screen_size, display_individual_classification=branch, display_graph_ensemble=False)
        fer_demo.update(fer)
        while fer_demo.is_running():
            fer_demo.show()
        fer_demo.quit()


def video(input_video_path, display, gradcam, output_csv_file, screen_size, device, fps, branch, no_plot):
    """
    This method receives the full path to a video file and recognizes
    facial expressions of the closets face in a frame-based approach.

    TODO: Write docstring.

    :param input_video_path:
    :param display:
    :param gradcam:
    :param output_csv_file:
    :param screen_size:
    :param device:
    :param fps:
    :param branch:
    :return:
    """
    fer_demo = None

    if not uimage.initialize_video_capture(input_video_path):
        raise RuntimeError("Error on initializing video capture." +
                           "\nCheck whether working versions of ffmpeg or gstreamer is installed." +
                           "\nSupported file format: MPEG-4 (*.mp4).")

    uimage.set_fps(fps)

    # Initialize screen
    if display:
        fer_demo = FERDemo(screen_size=screen_size, display_individual_classification=branch, display_graph_ensemble=(not no_plot))

    try:
        # Loop to process each frame from a VideoCapture object.
        while uimage.is_video_capture_open() and ((not display) or (display and fer_demo.is_running())):
            # Get a frame
            image = uimage.get_frame()

            fer = None if (image is None) else cvision.recognize_facial_expression(image, device)

            # Display blank screen if no face is detected, otherwise,
            # display detected faces and perceived facial expression labels
            if display:
                fer_demo.update(fer)
                fer_demo.show()

            # TODO: Implement
            if output_csv_file:
                pass

    except Exception as e:
        print("Error raised during video mode.")
        raise e
    finally:
        uimage.release_video_capture()
        fer_demo.quit()


def main():
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="selects a method among 'image', 'video' or 'webcam' to run ESR-9.",
                        type=str, choices=["image", "video", "webcam"])
    parser.add_argument("-d", "--display", help="displays the output of ESR-9.",
                        action="store_true")
    parser.add_argument("-g", "--gradcam", help="runs grad-CAM and displays the salience maps.",
                        action="store_true")
    parser.add_argument("-i", "--input", help="defines the full path to an image or video.",
                        type=str)
    parser.add_argument("-o", "--output", help="saves ESR-9's outputs in a CSV file defined in 'output' (ex. ./output.csv).",
                        type=str)
    parser.add_argument("-s", "--size",
                        help="defines the size of the window: \n1 - 1920 x 1080;\n2 - 1440 x 900;\n3 - 1024 x 768.",
                        type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("-c", "--cuda", help="runs on GPU.",
                        action="store_true")
    parser.add_argument("-w", "--webcam_id",
                        help="defines the webcam by 'id' to capture images in the webcam mode." +
                             "If none is selected, the default camera by the OS is used.",
                        type=int, default=-1)
    parser.add_argument("-f", "--fps", help="defines fps of videos and webcam captures.",
                        type=int, default=5)
    parser.add_argument("-b", "--branch", help="shows individual branch's classification if set true, otherwise," +
                                               "shows final ensemble's classification.",
                        action="store_true", default=False)
    parser.add_argument("-np", "--no_plot", help="do not display activation and (un)pleasant graph",
                        action="store_true", default=False)

    args = parser.parse_args()

    # Calls to main methods
    # TODO: Double-check args. Many are missing.
    if args.mode == "image":
        try:
            cvalidation.validate_image_video_mode_arguments(args)
            image(args.input, args.display, args.gradcam, args.output, args.size, args.cuda, args.branch)
        except RuntimeError as e:
            print(e)
    elif args.mode == "video":
        try:
            cvalidation.validate_image_video_mode_arguments(args)
            video(args.input, args.display, args.gradcam, args.output, args.size, args.cuda, args.fps, args.branch, args.no_plot)
        except RuntimeError as e:
            print(e)
    elif args.mode == "webcam":
        try:
            cvalidation.validate_webcam_mode_arguments(args)
            webcam(args.webcam_id, args.display, args.gradcam, args.output, args.size, args.cuda, args.fps, args.branch, args.no_plot)
        except RuntimeError as e:
            print(e)


if __name__ == "__main__":
    main()
