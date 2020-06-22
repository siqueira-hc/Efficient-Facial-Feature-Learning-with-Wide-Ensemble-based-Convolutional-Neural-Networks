import cv2
from controller import cvision

# Read an image
image = cv2.imread('./media/jackie.jpg', cv2.IMREAD_COLOR)

# Recognize a facial expression if a face is detected. The boolean argument set to False indicates that the process runs on CPU
fer = cvision.recognize_facial_expression(image, False, 1, False)

# Print list of emotions (individual classification from 9 convolutional branches and the ensemble classification)	
print(fer.list_emotion)
