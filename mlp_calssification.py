import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
# always check your version
print(torch.__version__)
import torch
import math
torch.device('mps')
# True
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('mps' if torch.backends.mps.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, num_classes=30):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4=nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512)
        )
        self.reLu1=nn.ReLU()
        self.fc1 = nn.Linear(512, 1024)
        self.reLu2=nn.ReLU()

        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out=self.layer4(out)
        out=self.reLu1(out)
        out = self.fc1(out)
        out=self.reLu2(out)
        out = self.fc2(out)


        return out

def built_model_cnn(CNN_model,learning_rate):
    num_classes=30
    # Set up the model, loss function, and optimizer
    model_CNN = CNN_model(num_classes=num_classes).to(device)
    loss_function_CNN = nn.CrossEntropyLoss()
    optimizer_CNN = optim.SGD(model_CNN.parameters(), lr=learning_rate,momentum=0.9)
    return model_CNN,loss_function_CNN,optimizer_CNN
class CNNWithDropout(CNN):
    def __init__(self, num_classes=30):
        super(CNNWithDropout, self).__init__()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        x = self.dropout(x)
        out=self.layer4(out)
        out=self.reLu1(out)
        out = self.fc1(out)
        out=self.reLu2(out)
        out = self.fc2(out)
        return out

 # load the classes
file = open("./comp5623m-artificial-intelligence/class.txt", "r")
contents=[]
for line in file.readlines():
    curLine=line.strip().split("\t")
    contents.append(curLine[:])
classes=[items[1] for items in contents]
# print('classes:',classes)



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
            if pred>1:
                break
            cam_extractor = SmoothGradCAMpp(model, target_layer='layer3.3')
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
            # Show the ordinary image using the default viewer on your system
            img = plt.imread(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()



# Load the saved model
learning_rate=0.01
model_cnn=CNNWithDropout(num_classes=30)
model_cnn.load_state_dict(torch.load('best_model_cnn.pth'))
print(type(model_cnn))
# model_cnn.to(device)
transform_cnn = transforms.Compose([
    # transforms.RandomHorizontalFlip(p=0.1),
    # transforms.RandomVerticalFlip(p=0.1),
    # transforms.RandomRotation(10),
    # transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1)),
    # transforms.ColorJitter(brightness=0.1,  saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

get_gradcam(model_cnn, 'cnn',image_paths, transform_cnn)