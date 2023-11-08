# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:20:45 2023

@author: HP
"""
import streamlit as st
import tensorflow as tf
import pickle
import torchvision
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageOps
import numpy as np

def load_model():
     model = torch.load('C:/Users/HP/Desktop/Sheeza/Cherry Tree app/CherryLeavesModel5.pth',map_location=torch.device('cpu'))
     return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.title("Cherry Classification")

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)
def import_(image_data, model):
           """resized_img = image_data.resize((2244,224))
           image = np.asarray(resized_img)
           img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
           img_reshape = img[np.newaxis,...]
           return img_reshape"""
           preprocess = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])
           input_data = preprocess(image)
           input_data = input_data.unsqueeze(0)  # Add a batch dimension
           return input_data
        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    im = import_(image, model)
    model.eval()  # Put the model in evaluation mode
    with torch.no_grad():
       prediction = model(im)
    class_names = ('healthy','powdery_mildew')
    score = tf.nn.softmax(prediction[0])
    st.write(prediction)
    st.write(score)
    st.write(class_names[np.argmax(score)])
    st.write(100 * np.max(score))
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)