# Efficient Facial Feature Learning with Wide Ensemble-based Convolutional Neural Networks
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-facial-feature-learning-with-wide/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=efficient-facial-feature-learning-with-wide)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-facial-feature-learning-with-wide/facial-expression-recognition-on-fer)](https://paperswithcode.com/sota/facial-expression-recognition-on-fer?p=efficient-facial-feature-learning-with-wide)

**This web page is on development as well as some features of our framework!**

This repository contains:
- Facial expression recognition framework.
- Introduction to Ensembles with Shared Representations.
- Implementation of an Ensemble with Shared Representations in PyTorch.
- Scripts of experiments conducted for the AAAI-2020 conference.
- [Our AAAI-2020 paper](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/Siqueira-AAAI_2020.pdf).
- [Our AAAI-2020 poster](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/Siqueira-AAAI_2020-Poster.pdf).

#### Updates
- **The Grad-CAM visualization algorithm has been implemented!** Click [here](#generating-saliency-maps-with-grad-cam) to learn more about this new feature.
- Face detection algorithm has been improved!
- The option to run facial expression recognition on GPU is now available. 

# Facial Expression Recognition Framework
![Example of the output of the framework in the video mode without a plot](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/terminator.gif)

The facial expression recognition framework can be started by running **main_esr9.py** followed by positional and optional arguments.

## Getting Started
### Installation
1. Install python 3.6.
2. Install dependencies by running:

```
pip install -r requirements.txt
```

Main python libraries used in our framework:
- matplotlib 3.0.3
- numpy 1.17.4
- opencv-python 4.1.2.30
- Pillow 5.0.0
- torch 1.0.0
- torchvision 0.2.1

### Features
The facial expression recognition framework has three main features:

1. Image: recognizes facial expressions in images.
2. Video: recognizes facial expressions in videos in a frame-based approach.
3. Webcam: connects to a webcam and recognizes facial expressions of the closest face detected by a face detection algorithm.

You can also import cvision and call the method **recognize_facial_expression** as follows:

```
import cv2
from controller import cvision

# Read an image
image = cv2.imread('./media/jackie.jpg', cv2.IMREAD_COLOR)

# Recognize a facial expression if a face is detected. The boolean argument set to False indicates that the process runs on CPU
fer = cvision.recognize_facial_expression(image, False)

# Print list of emotions (individual classification from 9 convolutional branches and the ensemble classification)	
print(fer.list_emotion)
```

#### Facial Expression Recognition in Images: Image Mode
To recognize a facial expression in images, run the following command:

```
python main_esr9.py image -i ./media/jackie.jpg -d -s 2
```

The argument **"image"** indicates that the input is an image. The location of the image is specified after **-i** while **-d** sets the display mode to true and **-s 2** sets the window size to 1440 x 900.

The framework should display the following image:

![Example of the output of the framework in the image mode](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/image_mode.png)

#### Generating Saliency Maps with Grad-CAM
You can also visualize regions in the image relevant for the classification of facial expression by adding -b -g as arguments:

```
python main_esr9.py image -i ./media/jackie.jpg -d -s 2 -b -g
```

The argument **-b** shows the classification of each branch and the argument **-g** generates saliency maps with **the Grad-CAM algorithm**.

The framework should display the following image:

![Example of the output of the framework in the image mode with Grad-CAM activated](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/grad_cam.png)

Zoom in with the mouse wheel for better visualization:

![Zoom-in image with Grad-CAM activated](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/grad_cam_zoom-in.png)

#### Facial Expression Recognition in Videos: Video Mode
To recognize a facial expression in videos, run the following command:


```
python main_esr9.py video -i ./media/big_bang.mp4 -d -f 5 -s 2
```

The argument **"video"** indicates that the input is a video. The location of the video is specified after **-i**. **-d** sets the display mode to true, **-f** defines the number of frames to be processed, and **-s 2** sets the window size to 1440 x 900.

Results should be displayed in a similar interface:

![Example of the output of the framework in the video mode](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/video_mode.png)

#### Recognizing Facial Expressions on the Fly: Webcam Mode
To recognize a facial expression in images captured from a webcam, run the following command:


```
python main_esr9.py webcam -d -s 2 -b
```

The argument **"webcam"** indicates the framework to capture images from a webcam. **-d** sets the display mode to true, **-s 2** sets the window size to 1440 x 900, and **-b** changes the default interface to show individual classification from each convolutional branch as follows:

![Example of the output of the framework in the webcam mode](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/webcam_mode.png)

### List of Arguments:
You can run python `main_esr9.py -h` or `python main_esr9.py -help` to display the help message with the list of arguments.

Positional arguments:
- **mode**:
	- Selects the running mode of the demo which are 'image', 'video' or 'webcam'.
	- Input values: {image, video, webcam}.

Optional arguments:
- **-h (--help)**:
	- Displays the help message.
- **-d (--display)**:
	- Displays an window with the input data on the left and the output data on the rigth (i.e., detected face, emotions, and affect values).
- **-i (--input)**:
	- Defines the full path to an image or video.
- **-s (--size)**:
	- Defines the size of the window:
		1. 1920 x 1080.
		2. 1440 x 900.
		3. 1024 x 768.
	- Input values: {1, 2, 3}.
- **-b (--branch)**:
	- Shows individual branch's classification.
- **-np (--no_plot)**:
	- Hides the graph of activation and (un)pleasant values.
- **-fd (--face_detection)**:
	- _**[On development]**_
	- Defines the face detection algorithm:
	    - 1. Optimized Dlib.
	    - 2. Standard Dlib (King, 2009).
	    - 3. Haar Cascade Classifiers (Viola and Jones, 2004).
	- _Warning: the chosen algorithm may affect performance._
- **-c (--cuda)**:
	- Runs facial expression recognition on GPU.
- **-w (--webcam)**:
	-  Defines the webcam to be used while the framework is running by 'id' when the webcam mode is selected. The default camera is used, if 'id' is not specified.
- **-f (--frames)**:
	-  Sets the number of frames to be processed for each 30 frames. The lower is the number, the faster is the process.
- **-o (--output)**:
	- _**[On development]**_
	- Saves ESR-9's outputs in a CSV file in the speficied location.
- **-g (--gradcam)**:
	- _**[On development]**_
	- Runs the grad-CAM algorithm and shows the saliency maps with respect to each convolutional branch.

### Citation:
If you found our framework and/or paper useful, please, consider citing us:
```
@InProceedings\{SMW20,
  author       = "Siqueira, Henrique and Magg, Sven and Wermter, Stefan",
  title        = "Efficient Facial Feature Learning with Wide Ensemble-based Convolutional Neural Networks",
  booktitle    = "The Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20)",
  pages        = "1--1",
  month        = "Feb",
  year         = "2020",
  url          = "https://www2.informatik.uni-hamburg.de/wtm/publications/2020/SMW20/SMW20.pdf"
}
```

# Ensembles with Shared Representations (ESRs)
_**[On development]**_

# Implementation of an ESR in PyTorch
_**[On development]**_

# Scripts of our AAAI-2020 experiments
_**[On development]**_

# Acknowledgements
This work has received funding from the European Union's Horizon 2020 research and innovation program under the Marie Skłodowska-Curie grant agreement No. 721619 for the SOCRATES project.

![Images of the EU flag and SOCRATES project](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/logo.png) 
