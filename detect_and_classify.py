import cv2
import numpy as np
from image_utils import *
import jetson.utils
import jetson.inference

net = jetson.inference.detectNet("ssd-mobilenet-v2")

if __name__ == "__main__":
    stream = cv2.VideoCapture('test_videos/1.mp4')
    
    while(stream.read()[0]):
        frame = stream.read()[1]

	#Convert from cv to cuda
        bgr_frame = jetson.utils.cudaFromNumpy(frame, isBGR=True)
        rgb_cuda_frame = jetson.utils.cudaAllocMapped(width=bgr_frame.width, height=bgr_frame.height, format='rgb8')
        jetson.utils.cudaConvertColor(bgr_frame, rgb_cuda_frame)

	#Perform Detection
        net.Detect(rgb_cuda_frame)

	#Convert from cuda back to cv
        bgr_cv_frame = jetson.utils.cudaAllocMapped(width=rgb_cuda_frame.width,height=rgb_cuda_frame.height, format='bgr8')
        jetson.utils.cudaConvertColor(rgb_cuda_frame, bgr_cv_frame)
        jetson.utils.cudaDeviceSynchronize()
        cv_frame = jetson.utils.cudaToNumpy(bgr_cv_frame)

        label_image(cv_frame, "Sunny Zurich", "uncertainty")

        cv2.imshow('Input', cv_frame)
        cv2.waitKey(1)
