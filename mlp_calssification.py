import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url

from PIL import Image
import matplotlib.pyplot as plt
# always check your version
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(torch.__version__)
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# print('Using device:', device)

import os
import torch.utils.data as data

class TinyImageNet30Dataset(data.Dataset):
    def __init__(self, root_dir,classes):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.classes = classes

        # Iterate over all classes
        for i, c in enumerate(classes):
            class_dir = os.path.join(root_dir, c)
            # print(i,c)
            if os.path.isdir(class_dir):
                # Add all images in this class directory to the dataset
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(i)



    def __getitem__(self, index):
        # Load the image at the given index
        image_path = self.image_paths[index]
        # print(image_path)
        # Use PIL for image loading
        image = Image.open(image_path).convert('RGB')

        # Apply any specified image transformations
        tensor_image = transforms.ToTensor()(image)

        # Retrieve the label for this image
        label = self.labels[index]

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return tensor_image, label

    def __len__(self):
        # Return the previously computed number of images
        return len(self.image_paths)
from torch.utils.data import DataLoader, random_split

# load the dataset
train_set_root='./comp5623m-artificial-intelligence/train_set/train_set/train_set'

# load the classes
file = open("./comp5623m-artificial-intelligence/class.txt", "r")
contents=[]
for line in file.readlines():
    curLine=line.strip().split("\t")
    contents.append(curLine[:])
classes=[items[1] for items in contents]
print('classes:',classes)
# Create the dataset
dataset=TinyImageNet30Dataset(root_dir=train_set_root,classes=classes)
dataset_load=DataLoader(dataset)


# TO COMPLETE
# define a MLP Model class
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_units, hidden_units, output_units):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_units)
        self.activation_fn = nn.ReLU()

    def forward(self, x):
        x=self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x

# define a MLP model using GPU
input_units = 64  # assuming images are flattened to a 784-dimensional vector
hidden_units = 256  # number of hidden units
output_units = 30   # number of output units
activation_fn = nn.ReLU()
mlp_model = MLP(input_units, hidden_units, output_units)#.to(device)

# define a loss function and an optimizer
loss_fn_mlp = nn.CrossEntropyLoss()
optimizer_mlp=optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)


# Set seed for reproducibility
torch.manual_seed(0)

# Split the dataset into train and validation sets
test_ratio=0.2
# Split train_dataset into train and validation sets
train_size = int(0.8 * len(dataset_load.dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# Create DataLoader for train, validation, and test sets
train_loader = DataLoader(train_dataset)
val_loader = DataLoader(val_dataset)
print('train size:', len(train_dataset))
print('val size:', len(val_dataset))


# Define top-*k* accuracy
def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



# Train the model
train_losses = []
val_losses = []
train_accs = []
val_accs = []
num_epochs=50

for epoch in range(num_epochs):
    print('starting epoch {}'.format(epoch))
    # Train the model
    mlp_model.train()
    print('     training the MLP model')
    running_loss = 0.0
    running_acc = 0.0
    for inputs, targets in train_loader:
        print('len(inputs):',len(inputs))
        print('len(targets):',len(targets))
        inputs, targets = inputs, targets #.to(device)
        optimizer_mlp.zero_grad()
        outputs = mlp_model.forward(inputs)
        print('outputs:',len(outputs))
        loss = loss_fn_mlp(outputs, targets)
        print('loss:',loss)
        loss.backward()
        print('loss.backward():',loss.backward())
        optimizer_mlp.step()
        acc = topk_accuracy(outputs, targets, topk=(1,))[0]
        print('acc:',topk_accuracy(outputs, targets, topk=(1,)))
        running_loss += loss.item() * inputs.size(0)
        running_acc += acc.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    print('     trained the MLP model!')
    # Validate the model
    # Evaluate the model on the validation set
    mlp_model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = mlp_model(inputs)
            loss = loss_fn_mlp(outputs, targets)
            acc = topk_accuracy(outputs, targets, topk=(1,))#[0]
            running_loss += loss.item() * inputs.size(0)
            running_acc += acc.item() * inputs.size(0)
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_acc / len(val_loader.dataset)
    val_losses.append(epoch_loss)
    val_accs.append(epoch_acc)

    # Print the epoch loss and accuracy
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}")

    # Check for early stopping
    if epoch > 0 and val_losses[-1] > val_losses[-2]:
        print("Early stopping at "+str(epoch))
        break


# Your graph
# Plot the training and validation loss and accuracy over epochs
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(train_accs, label="Train top 1 Acc")
plt.plot(val_accs, label="Val top 1 Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()