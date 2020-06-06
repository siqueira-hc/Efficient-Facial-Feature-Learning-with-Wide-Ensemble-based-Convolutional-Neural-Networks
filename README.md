# Efficient Facial Feature Learning with Wide Ensemble-based Convolutional Neural Networks
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-facial-feature-learning-with-wide/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=efficient-facial-feature-learning-with-wide)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-facial-feature-learning-with-wide/facial-expression-recognition-on-fer)](https://paperswithcode.com/sota/facial-expression-recognition-on-fer?p=efficient-facial-feature-learning-with-wide)

This repository contains:
- Facial expression recognition framework.
- Introduction to Ensembles with Shared Representations.
- Implementation of an Ensemble with Shared Representations in PyTorch.
- Scripts of experiments conducted for the AAAI-2020 conference.
- [Our AAAI-2020 paper](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/Siqueira-AAAI_2020.pdf).
- [Our AAAI-2020 poster](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/Siqueira-AAAI_2020-Poster.pdf).

#### Updates

- **Training scripts are now available.** They do not run out-of-the-box. Check out [this section](#scripts-of-our-aaai-2020-experiments) for guidelines.
- **You can now save predictions to a CSV file!** To test the new feature, run the following command:
```
python main_esr9.py video -i ./media/terminator.mp4 -d -f 5 -s 3 -o ./test/
```
- **The Grad-CAM visualization algorithm has been implemented!** Click [here](#generating-saliency-maps-with-grad-cam) to learn more about this new feature.
- Face detection algorithm has been improved!
- The option to run facial expression recognition on GPU is now available. 

# Facial Expression Recognition Framework
![Example of the output of the framework in the video mode without a plot](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/terminator.gif)

The facial expression recognition framework can be started by running **main_esr9.py** followed by positional and optional arguments.

## Getting Started
### Installation
1. Install python 3.6.
2. (Optional but recommended) Create a virtual environment for the installation and activate it (using Anaconda Prompt):
```
conda create --name your_env_name python=3.6
conda activate your_env_name
```

2. Change directories to wherever this project was installed. Then, install dependencies by running:

```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

Main python libraries used in our framework:
- matplotlib 3.2.1
- numpy 1.18.5
- opencv-python 4.2.0.34
- Pillow 7.1.2
- torch 1.5.0+cpu
- torchvision 0.6.0+cpu

Note: if your system has CUDA, you may get better performance by installing the GPU-enabled version of torch and torchvision instead. But regardless, the CPU version should still work. If you want to do that, go to pytorch.org to determine which version of torch and torchvision you should install. Remember to delete the lines
```
torch==1.5.0+cpu
torchvision==0.6.0+cpu
```
in requirements.txt before running the ```pip install``` command above. Then, run the command that pytorch.org gave you to install GPU-enabled torch and torchvision.


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
python main_esr9.py webcam -w 0 -d -s 2 -b
```

The argument **"webcam"** indicates the framework to capture images from a webcam. **-w 0** tells the program to use camera 0 (usually the default, corresponds to the built-in webcam on your machine), **-d** sets the display mode to true, **-s 2** sets the window size to 1440 x 900, and **-b** changes the default interface to show individual classification from each convolutional branch as follows:

![Example of the output of the framework in the webcam mode](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/webcam_mode.png)

### List of Arguments:
You can run python `main_esr9.py -h` or `python main_esr9.py -help` to display the help message with the list of arguments.

Positional arguments:
- **mode**:
	- Select the running mode of the demo which are 'image', 'video' or 'webcam'.
	- Input values: {image, video, webcam}.

Optional arguments:
- **-h (--help)**:
	- Display the help message.
- **-d (--display)**:
	- Display an window with the input data on the left and the output data on the rigth (i.e., detected face, emotions, and affect values).
- **-i (--input)**:
	- Define the full path to an image or video.
- **-s (--size)**:
	- Define the size of the window:
		1. 1920 x 1080.
		2. 1440 x 900.
		3. 1024 x 768.
	- Input values: {1, 2, 3}.
- **-b (--branch)**:
	- Show individual branch's classification.
- **-np (--no_plot)**:
	- Hide the graph of activation and (un)pleasant values.
- **-fd (--face_detection)**:
	- Define the face detection algorithm:
	    - 1. Optimized Dlib.
	    - 2. Standard Dlib (King, 2009).
	    - 3. Haar Cascade Classifiers (Viola and Jones, 2004).
	- _Warning: the chosen algorithm may affect performance._
- **-c (--cuda)**:
	- Run facial expression recognition on GPU.
- **-w (--webcam)**:
	-  Define the webcam to be used while the framework is running by 'id' when the webcam mode is selected. The default camera is used, if 'id' is not specified.
- **-f (--frames)**:
	-  Set the number of frames to be processed for each 30 frames. The lower is the number, the faster is the process.
- **-o (--output)**:
	- Create and write ESR-9's outputs to a CSV file.
	- The file is saved in a folder defined by this argument (ex. '-o ./' saves the file with the same name as the input file in the working directory).
- **-g (--gradcam)**:
	- Run the grad-CAM algorithm and shows the saliency maps with respect to each convolutional branch.

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
![Ensembles with Shared Representations](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/architecture.png)

Ensembles with shared representations exploit the fundamental properties of convolutional networks. A convolutional layer learns local patterns from the previous layer by convolving small filters over its input feature space. Thus, the patterns learned by convolutional layers are translation-invariant. Another property is the capability to learn spatial hierarchies of patterns by stacking multiple convolutional layers. Consider the task of automatic facial expression recognition. Early layers learn simple and local visual patterns such as oriented lines, edges, and colors. Subsequent layers hierarchically combine local patterns from previous layers into increasingly complex concepts such as nose, mouth, and eyes. The level of abstraction increases as you go deeper into the network until the point where feature maps are no longer visually interpretable. Finally, the last layer encodes these representations into semantic concepts, for instance, concepts of emotion.

These properties are the foundations of ESRs and play a crucial role in reducing redundancy of visual features in the ensemble. An ESR consists of two building blocks. (1) The base of the network (sequence of blocks on the left in the Figure above) is an array of convolutional layers for low- and middle-level feature learning. (2) These informative features are then shared with independent convolutional branches (set of sequences of blocks on the right in the Figure above) that constitute the ensemble. From this point, each branch can learn distinctive features while competing for a common resource - the shared layers.

There are a few examples of ESRs in this repository. Check out ./model/ml/esr_9.py for a simple implementation of ESR-9 for discrete emotion and continuous affect perception.

In the scripts main_ck_plus.py, main_affectnet_discrete.py, main_affectnet_continuous.py, and main_fer_plus.py, you will find different ways to train or to fine-tune ESRs.

# Scripts of our AAAI-2020 experiments
## Training an ESR-4 on the Extended Cohn-Kanade Dataset
To train ESR-4 (ESR with four convolutional branches. In our paper, it is referred as ESR-4 Lvl. 3 Frozen Layers.) on the Extended Cohn-Kanade dataset, run the script main_ck_plus.py. However, this script does not run out-of-the-box. To be able to run the main_ck_plus.py script, one shall download and organized the dataset into the following structure:

```
Cohn-Kanade - Extended/
    cohn-kanade-images/
        S005/
            001/
                S005_001_00000001.png
                ...
            ...
        ...        
    Emotion/
            S005/
                001/
                    S005_001_00000011_emotion.txt
                    ...
                ...
            ...
```

The images in the folders must be pre-processed including cropping the face and rescaling to 96x96 pixels. For more details about the pre-processing and experiments on the Extended Cohn-Kanade, please, read [our AAAI-2020 paper](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/Siqueira-AAAI_2020.pdf).

After that, set the experimental variables including the base path to the dataset (base_path_to_dataset = "[...]/Cohn-Kanade - Extended/").

The Extended Cohn-Kanade dataset is available at [http://www.jeffcohn.net/Resources/](http://www.jeffcohn.net/Resources/).

## Discrete Emotion Perception: Training ESR-9 from scratch on the AffectNet Dataset
To train ESR-9 on the AffectNet dataset for discrete emotion perception, run the script main_affectnet_discrete.py. However, this script does not run out-of-the-box. To be able to run the main_affectnet_discrete.py script, one shall download and organized the dataset into the following structure:

```
AffectNet/    
    Training_Labeled/
        0/
        1/
        ...
        n/
    Training_Unlabeled/
        0/
        1/
        ...
        n/
    Validation/
        0/
        1/
        ...
        n/
```

The folder 0/, 1/, ..., /n contains up to 500 images from the AffectNet after pre-processing. To pre-process the images and organize them into the above structure, call the method pre_process_affect_net(base_path_to_images, base_path_to_annotations) from ./model/utils/udata.py. The images will be cropped (to get the face only), re-scaled to 96x96 pixels, and renamed to follow the pattern "[id]_[emotion_idx]_[valence times 1000]_[arousal times 1000].jpg".

After that, set the experimental variables including the base path to the dataset (base_path_to_dataset = "[...]/AffectNet/").

The AffectNet dataset is available at [http://mohammadmahoor.com/affectnet/](http://mohammadmahoor.com/affectnet/).

## Continuous Affect Perception: Fine-tuning ESR-9 on the AffectNet Dataset 
To train ESR-9 on the AffectNet dataset for continuous affect perception, run the script main_affectnet_continuous.py. However, this script does not run out-of-the-box. To be able to run the main_affectnet_continuous.py script, one shall follow the instruction described in [the previous section](#discrete-emotion-perception-training-esr-9-from-scratch-on-the-affectNet-dataset).

The AffectNet dataset is available at [http://mohammadmahoor.com/affectnet/](http://mohammadmahoor.com/affectnet/).

## Fine-tuning ESR-9 on the FER+ Dataset
To fine-tune ESR-9 on the FER+ dataset, run the script main_fer_plus.py. However, this script does not run out-of-the-box. To be able to run the main_fer_plus.py script, one shall download and organized the dataset into the following structure:

```
FER_2013/
    Dataset/
        Images/
            FER2013Train/
            FER2013Valid/
            FER2013Test/
        Labels/
            FER2013Train/
            FER2013Valid/
            FER2013Test/
```

After that, set the experimental variables including the base path to the dataset (base_path_to_dataset = "[...]/FER_2013/Dataset/").

The FER 2013 dataset is available at [https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
Our experiments used the FER+ labels and they are available at [https://github.com/microsoft/FERPlus](https://github.com/microsoft/FERPlus).

# Acknowledgements
This work has received funding from the European Union's Horizon 2020 research and innovation program under the Marie Skłodowska-Curie grant agreement No. 721619 for the SOCRATES project.

![Images of the EU flag and SOCRATES project](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/media/logo.png) 
