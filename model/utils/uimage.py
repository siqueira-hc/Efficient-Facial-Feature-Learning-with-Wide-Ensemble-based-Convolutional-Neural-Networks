#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements image processing methods.
"""

__author__ = "Henrique Siqueira"
__email__ = "siqueira.hc@outlook.com"
__license__ = "MIT license"
__version__ = "1.0"

# Standard Libraries
import os

# External Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


# Image I/O >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class CVVideo(MiddlewareCommunicator):
    CAP_PROP_FRAME_WIDTH = 640
    CAP_PROP_FRAME_HEIGHT = 480

    def __init__(self, fps=20, max_fps=30, cap=None):
        super().__init__()
        self.max_fps = max_fps
        self.fps = fps
        self.cap = cap

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "CVVideo", "/esr9/cam_fps", should_wait=True)
    def set_fps(self, fps):
        self.fps = fps
        return fps,

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "CVVideo", "/esr9/cam_chk", should_wait=True)
    def is_video_capture_open(self):
        if self.cap is None:
            return False,
        else:
            return self.cap.isOpened(),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "CVVideo", "/esr9/cam_ini", should_wait=True)
    def initialize_video_capture(self, source, img_width=CAP_PROP_FRAME_WIDTH, img_height=CAP_PROP_FRAME_HEIGHT):

        # If cap is not none, it re-initialize video capture with the new video file
        if not (self.cap is None):
            self.cap.release()
            self.cap = None

        # Read the file
        try:
            self.cap = cv2.VideoCapture(source)
            if img_width > 0 and img_height > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
        except Exception as e:
            self.cap = None
            print("Error on trying to read the following file as video: {}".format(source))
            print("Please, check if the file exists, is an image and is not corrupted.")
            print("Supported file format: MPEG-4 (*.mp4).")
            print("Check whether working versions of ffmpeg or gstreamer is installed.")
            raise e

        return not (self.cap is None),

    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "CVVideo", "/esr9/cam_cls", should_wait=True)
    def release_video_capture(self):
        try:
            self.cap.release()
        except Exception as e:
            print(e)
        finally:
            self.cap = None

        return self.cap is None,

    @MiddlewareCommunicator.register("Image", DEFAULT_COMMUNICATOR, "CVVideo", "/esr9/cam_feed",
                                     width="$img_width", height="$img_height", rgb=True, queue_size=10)
    @MiddlewareCommunicator.register("NativeObject", DEFAULT_COMMUNICATOR, "CVVideo", "/esr9/cam_sec", should_wait=True)
    def get_frame(self, img_width=CAP_PROP_FRAME_WIDTH, img_height=CAP_PROP_FRAME_HEIGHT):
        """
        Get a frame from a video file.

        :return: (ndarray, float) (Loaded frame, time in seconds).
        """

        to_return_frame = None

        if self.cap is None:
            print("Error on getting frame. cv2.VideoCapture is not initialized.")
        else:
            try:
                if self.cap.isOpened():
                    # Skip frames
                    for i in range(int(self.max_fps / self.fps)):
                        self.cap.grab()

                    is_valid_frame, to_return_frame = self.cap.retrieve()

                    if not is_valid_frame:
                        to_return_frame = None
            except Exception as e:
                print("Error on getting a frame. Please, double-check if the video file is not corrupted.")
                print("Supported file format: MPEG-4 (*.mp4).")
                print("Check whether working versions of ffmpeg or gstreamer is installed.")
                raise e

        return to_return_frame, (self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)


def read(path_to_image, convert_to_grey_scale=False):
    """
    Reads the file as an image.
    :param path_to_image: (string)
    :param convert_to_grey_scale: (bool) opens an image and converts it to a 2d greyscale image.
    :return: (ndarray) 3d (channels last) or 2d image array.
    """

    loaded_image = None
    exception = None

    # Read the file
    try:
        if convert_to_grey_scale:
            loaded_image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
        else:
            loaded_image = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
    except Exception as e:
        loaded_image = None
        exception = e

    # Check if the file has been successfully read as an image
    if loaded_image is None:
        print("Error on trying to read the following file as an image: {}".format(path_to_image))
        print("Please, check if the file exists, is an image and is not corrupted.")
        print("Supported file formats: JPEG (*.jpeg and *.jpg) and Portable Network Graphics (*.png).")

        if exception is None:
            raise RuntimeError("Unable to read the file (unknown error:).")
        else:
            raise exception

    return loaded_image

def write(image, file_path, file_name):
    full_path = os.path.join(file_path, file_name)

    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    cv2.imwrite(full_path, image)

    print("Image successfully saved at: %s" % full_path)

# Image I/O <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# Color conversion methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def convert_grey_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def convert_bgr_to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_rgb_to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def convert_rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Color conversion methods <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Drawing methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def draw_rectangle(image, initial_coordinates, final_coordinates, color=(0, 255, 0), thickness=2):
    cv2.rectangle(image, initial_coordinates, final_coordinates, color, thickness)

def draw_horizontal_bar(image, val, max, initial_coordinates, final_coordinates, thickness, color=(0, 255, 0)):
    x_length = final_coordinates[0] - initial_coordinates[0]
    value_coordinates = (int(initial_coordinates[0] + ((x_length * val) / max)), final_coordinates[1])

    cv2.rectangle(image, initial_coordinates, final_coordinates, color, thickness)
    cv2.rectangle(image, initial_coordinates, value_coordinates, color, cv2.FILLED)

def draw_graph(image, x, y, initial_coordinates, samples, text_x, text_y, color_x, color_y, thickness, offset, font_size, grid_color, size):
    # Params
    plt.rcParams["figure.figsize"] = size
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams.update({"font.size": font_size})

    # Initialization
    fig = plt.figure()

    # Sampling
    len_x = np.minimum(len(x), samples)
    z = np.arange(len_x)
    x = np.array(x)[-samples:]
    y = np.array(y)[-samples:]

    # Data
    plt.plot(z, np.zeros(len_x), color=(0, 0, 0), linewidth=1.5)
    plt.plot(z, x, label=text_x, color=(np.array(color_x) / 255.), linewidth=thickness)
    plt.plot(z, y, label=text_y, color=(np.array(color_y) / 255.), linewidth=thickness)

    # Axises
    plt.ylim([-1.0, 1.0])
    plt.yticks(np.arange(-1.0, 1.1, 0.25))
    plt.xlim([0, len_x])
    plt.xticks(np.arange(0, len_x, 1))

    # Grid
    ax = plt.gca()
    ax.grid(color=(np.array(grid_color) / 255.), linestyle="--")
    ax.spines["top"].set_color((np.array(grid_color) / 255.))
    ax.spines["top"].set_linestyle("--")
    ax.spines["bottom"].set_color("#b5b5b5ff")
    ax.spines["bottom"].set_linestyle("--")
    ax.spines["right"].set_color("#b5b5b5ff")
    ax.spines["right"].set_linestyle("--")
    ax.spines["left"].set_color("#b5b5b5ff")
    ax.spines["left"].set_linestyle("--")
    ax.spines["right"].set_visible(False)

    # Legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=False, shadow=False, ncol=5)

    # From plt to ndarray
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    # Offset (right)
    data = data[:, :-offset, :]

    # Draw to image
    image[initial_coordinates[0]:initial_coordinates[0] + data.shape[0], initial_coordinates[1]:initial_coordinates[1] + data.shape[1], :] = data[:]

def draw_text(image, text, initial_coordinates, color=(0, 255, 0), scale=1, thickness=1):
    cv2.putText(image, text, (int(initial_coordinates[0]), int(initial_coordinates[1])), cv2.FONT_HERSHEY_COMPLEX,
                fontScale=scale, color=color, thickness=thickness)

def draw_image(image, image_to_draw, initial_coordinates):
    image[int(initial_coordinates[0]):int(initial_coordinates[0]) + image_to_draw.shape[0], int(initial_coordinates[1]):int(initial_coordinates[1]) + image_to_draw.shape[1], :] = image_to_draw

# Drawing methods <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Transformation methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def resize(image, output_size=None, f=None):
    try:
        if f is None:
            return cv2.resize(image, output_size)
        else:
            return cv2.resize(image, output_size, fx=f, fy=f)
    except cv2.Error:
        return image

def crop_rectangle(image, initial_coordinates, final_coordinates, channels_last=True):
    if channels_last:
        return image[initial_coordinates[1]:final_coordinates[1], initial_coordinates[0]:final_coordinates[0], :]
    else:
        return image[:, initial_coordinates[1]:final_coordinates[1], initial_coordinates[0]:final_coordinates[0]]

# Transformation methods <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Other methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def blur(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

def superimpose(img_1, img_2, w_1=0.35, w_2=0.65, gamma=0):
    # Convert tensor to numpy, resize to the input size, cast to uint8, and superimpose img_1 on img_2
    saliency_map = resize(img_1.cpu().detach().numpy(), output_size=(img_2.shape[1], img_2.shape[0]))
    saliency_map = cv2.applyColorMap(np.clip(saliency_map * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(saliency_map, w_1, img_2, w_2, gamma)

# Other methods <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
