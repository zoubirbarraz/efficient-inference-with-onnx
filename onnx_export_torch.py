#### Example for a convnext-tiny model
# imports 
import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device = cuda if available else cpu

# model definition
class ConvNext_tiny_pretrained(nn.Module): # simple implementation of the ConvNext_tiny
    def __init__(self, num_classes=2):
        super(ConvNext_tiny_pretrained, self).__init__()
        self.model = models.convnext.convnext_tiny(weights='DEFAULT') # importing the model with pretrained weights
        self.model.classifier[-1] = nn.Linear(768, num_classes) # changing the last layer to output num_classes

    def forward(self, x):
        return self.model(x) # forward pass 
    
#  Get the pre-trained model from path and load the weights to the model 
model_path = 'path/to/your/model.pth' # path to the model
model = ConvNext_tiny_pretrained() # creating the model
model.to(device) # moving the model to the device
model.load_state_dict(torch.load(model_path)) # loading the model weights
model.eval() # setting the model to evaluation mode

# set the input size used for training
input_size = (3, 224, 224) # input size of the model
input_tensor = torch.rand(1, *input_size) # creating a random input tensor
input_tensor = input_tensor.to(device) # moving the input tensor to the device (same as the model)

# Export the model to ONNX
torch.onnx.export(model, input_tensor, 'ConvNext_tiny_pretrained.onnx', verbose=True) # exporting the model to ONNX format

# Running Inference using the ONNX model

# imports
import onnx, onnxruntime
import onnx.numpy_helper
import numpy as np
import cv2
import glob

# onnx model path
model_onnx_path = 'ConvNext_tiny_pretrained.onnx'
# runtime initialization
ort_session = onnxruntime.InferenceSession(model_onnx_path, ['CPUExecutionProvider']) # use 'CUDAExecutionProvider' for CPU

# preparing the images for inference:
def preprocess_images(path): # path to folder containing images
    images = []
    for image_path in glob.glob(path + '/*.jpg'): # read all jpg images in the folder
        image = cv2.imread(image_path) # read the image
        image = cv2.resize(image, (224, 224)) # resize the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert the image to RGB , because opencv reads the image in BGR format
        image = np.transpose(image, (2, 0, 1)) # change the image format to CxHxW (channels first)
        image = np.expand_dims(image, axis=0) # add a batch dimension to the image (1, C, H, W), i.e one image per batch
        images.append(image) # append the image to the list
    return np.array(images) # return the list of images as a numpy array

# softmax function to get the probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# function to run inference
def run_inference(image, ort_session):
    input_name = ort_session.get_inputs()[0].name # get the input name of the model 
    output_name = ort_session.get_outputs()[0].name # get the output name of the model
    result = ort_session.run([output_name], {input_name: image}) # run the inference on the image 
    probabilities = softmax(result[0][0]) # apply softmax to the output to get the probabilities
    return probabilities

#define output class names as in the pre-trained model 
class_names = ['class_1', 'class_2', '...'] # class names

# run inference on the images
images = preprocess_images('path/to/your/images') # preprocess the images
for image in images:
    probabilities = run_inference(image, ort_session) # run the inference on the image
    class_idx = np.argmax(probabilities) # get the index of the class with the highest probability
    class_name = class_names[class_idx] # get the class name
    print(f'Class: {class_name}, Probability: {probabilities[class_idx]}') # print the class name and probability
