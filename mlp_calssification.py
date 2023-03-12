import math
import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url
# always check your version
print(torch.__version__)
import torch
import math
torch.device('mps')
# True
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('mps' if torch.backends.mps.is_available() else 'cpu')

from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.models as models
# Load pretrained AlexNet
model_alexnet = models.alexnet(pretrained=True)
# Your model changes here - also print trainable parameters
import copy
# No frozen layers
for param in model_alexnet.parameters():
    param.requires_grad = True
# Print trainable parameters
for name, param in model_alexnet.named_parameters():
    if param.requires_grad:
        print(name)
# Copy the model
model_alexnet_c1=copy.deepcopy(model_alexnet)

# Your changes here - also print trainable parameters



 # load the classes
file = open("./comp5623m-artificial-intelligence/class.txt", "r")
contents=[]
for line in file.readlines():
    curLine=line.strip().split("\t")
    contents.append(curLine[:])
classes=[items[1] for items in contents]
# print('classes:',classes)






from torchvision.io import read_image
# Your code here!
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.utils import overlay_mask


# # load CNN model
# model_cnn=CNN(num_classes=30)
# model_cnn.load_state_dict(torch.load('best_model_cnn.pth'))
# print(type(model_cnn))
# #
# transform_cnn= transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize(0.5,0.5)])




image_paths=[]
labels=[]
root_dir='./comp5623m-artificial-intelligence/train_set/train_set/train_set'
# load images_cnn_correct
# Iterate over all classes
for i, c in enumerate(classes):
    class_dir = os.path.join(root_dir, c)
    # print(i,c)
    if os.path.isdir(class_dir):
        # Add all images in this class directory to the dataset
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image_paths.append(image_path)
            labels.append(i)
print(len(image_paths))

def get_gradcam(model,model_name, image_paths, transform):

    pred=0
    model.eval()
    # Loop over the images
    for i, image_path in enumerate(image_paths):
        # Load the image and preprocess it
        image = Image.open(image_path).convert('RGB')
        preprocess = transform
        image_tensor= preprocess(image)
        input_tensor = image_tensor.unsqueeze(0)
        # Get the predicted class index
        output = model(input_tensor)
        # Retrieve the CAM by passing the class index and the model output

        _, predicted = torch.max(output.data, 1)

        if predicted.item()==labels[i] and model_name == 'cnn':
            print('predicted class is: ', predicted.item())
            print('true class is: ', labels[i])
            print('correctly classified')
            pred+=1
            if pred>4:
                break
            # cam(model, input_tensor, image_tensor)

        if predicted.item()==labels[i] and model_name == 'alexnet':
            print('predicted class is: ', predicted.item())
            print('true class is: ', labels[i])
            print('image path is: ', image_path)
            print('incorrectly classified')
            pred+=1
            if pred>4:
                break
            cam_extractor = SmoothGradCAMpp(model, target_layer='features.12')
            # Preprocess your data and feed it to the model
            out = model(input_tensor)
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

            # Resize the CAM and overlay it
            result = overlay_mask(to_pil_image(image_tensor), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            # Display it
            plt.imshow(result);
            plt.axis('off');
            plt.tight_layout();
            plt.show()


# # load model_alex
model_alex= model_alexnet
model_alex.classifier[6] = torch.nn.Linear(4096, 30)
model_alex.load_state_dict(torch.load('best_model_alex.pth'))
model_alex
print(type(model_alex))
transform_alex= transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(0.5,0.5)])



get_gradcam(model_alex, 'alexnet',image_paths, transform_alex)