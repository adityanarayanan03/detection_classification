import cv2
import numpy as np
from image_utils import *

if __name__ == "__main__":
    stream = cv2.VideoCapture('test_videos/1.mp4')
    
    while(stream.read()[0]):
        frame = stream.read()[1]

        label_image(frame, "Sunny Zurich", 80)

        cv2.imshow('Input', frame)
        cv2.waitKey(1)