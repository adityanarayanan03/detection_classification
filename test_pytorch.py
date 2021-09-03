import numpy as np
import cv2

import torch

from PIL import Image

from image_utils import *
from model_config import *


if __name__ == "__main__":

    #Create model architecture
    model = make_ml_model()

    #Load the weights as a state dictionary, from specified file
    filename = 'models/ml_model_0'
    state_dict = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    #Put the model into inference mode
    model.eval()


    #Define a data stream from one of the test videos
    stream = cv2.VideoCapture('test_videos/1.mp4')
    
    while(stream.read()[0]): #While stream is running
        
        frame = stream.read()[1] #Slice video into single frame

        #Convert frame from cv2 to a PIL image
        color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_converted)

        #Apply the transform in image_utils to get a pytorch tensor
        img_t = transform(pil_image)

        #Run the image through the ML model
        batch_t = torch.unsqueeze(img_t, 0)
        out = model(batch_t)
        result = torch.argmax(out)

        #Label image and display image
        label_image(frame, f"{result}", "No UNC :(")
        cv2.imshow("Detected and Classified Frame", frame)
        cv2.waitKey(1)