import os
import logging
import warnings

# disable TensorFlow warnings & absl logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

logging.getLogger('tensorflow').setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')


import tensorflow as tf
from torchvision import transforms
from ultralytics import YOLO 


def load_accident_model():
    
    model = YOLO('C:/Users/rahaf/httpsgithub.com/Tanabbah_capstone/best.pt')  
    return model


# Preprocess the uploaded image & preprocess
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  
    ])
    return preprocess(image).unsqueeze(0) 


# detect accident on processed image
def detect_accidents(model, image):
    
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(next(model.model.parameters()).dtype)  
    output = model.predict(image_tensor)
    return output







