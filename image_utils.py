import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import cv2
import numpy as np
from torchvision import transforms


#Defines a transform from PIL image into a pytorch tensor
transform = transforms.Compose([            #[1]
        transforms.Resize(256),                    #[2]
        #transforms.CenterCrop(224),                #[3]
        transforms.ToTensor(),                     #[4]
        transforms.Normalize(                      #[5]
        mean=[0.485, 0.456, 0.406],                #[6]
        std=[0.229, 0.224, 0.225]                  #[7]
        )
        ])


def label_image(image, classification, unc = None, display_image=True):
    '''
    Annotates an image with a classification and uncertainty score

    Keyword Arguments:
    image -- Numpy array or PIL image, image to annotate
    classification -- String, label for image
    unc -- uncertainty score for label (default None)
    display_image -- Bool, create cv2 window to show image or not (default True)
    '''
    
    #Robustify the function because python is dumb and doesn't allow overloading
    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
    
    draw = PIL.ImageDraw.Draw(image)

    draw_string = f"{classification}, {unc}"

    font = PIL.ImageFont.truetype("fonts/cmunss.ttf", 45) 
    draw.text((10,10), draw_string, font=font)

    image_out= np.array(image)

    if display_image:
        cv2.imshow("Model Output", image_out)
    
    return image_out