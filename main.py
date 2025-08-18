import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from time import sleep
import torch
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import TBClassifier

#const
INFECTED_PATH = "./data/Tuberculosis/"
NORMAL_PATH = "./data/Normal/"
BASE_PATH = "./data/"

TARGET_SIZE = (224, 224)
VALIDATION_SPLIT_RATIO = 0.15
THE_SEED = 42

# print(f"Number of infected images: {len(os.listdir(INFECTED_PATH))}") # 700 img
# print(f"Number of normal images: {len(os.listdir(NORMAL_PATH))}") # 3500 img


'''
row1 : normal 1     normal 2      normal 3
row2 : infected 1     infected 2      infected 3
'''

def show_data(category, n=5, rgb=True):
    fig, axes = plt.subplots(1, n, figsize=(15,5))
    for i, fname in enumerate(os.listdir(category)[:n]):
        file_path = category + fname
        print(file_path)
        with Image.open(file_path) as img:
            if not rgb:
                img = img.convert("L")
            img_np=  np.array(img)
            axes[i].imshow(img_np)
            axes[i].axis('off')
            # img.close()

    plt.show(block=False)
    plt.pause(5)
    plt.close(fig)

# show_data(NORMAL_PATH)
# show_data(INFECTED_PATH)

#preprocessing
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])

val_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.CenterCrop(TARGET_SIZE),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])

# print("im running")

full_train_dataset = datasets.ImageFolder(root=BASE_PATH, transform=train_transform)      
full_val_dataset = datasets.ImageFolder(root=BASE_PATH, transform=val_transform)

# print(full_train_dataset.targets)

all_labels = np.array(full_train_dataset.targets)
all_indices = np.arange(len(full_train_dataset))

print(f"label: {all_labels}\nindicies: {all_indices}")

train_indices, val_indices, _, _ = train_test_split(
    all_indices, # Indices of the dataset
    all_labels,  # Labels corresponding to the indices (for stratification)
    test_size= VALIDATION_SPLIT_RATIO,
    stratify=all_labels, # This is the key argument for stratification
    random_state=THE_SEED # For reproducibility
)



"""
indicies labels
0          0 test
1          1 test
2          1 train
...
4198       1 train
4199       0 train

"""


"""
x    y
ade  1 train
bel  0 trsin
imi  1 train
jam  0 test
dnam 0 test
"""



datasets_size = len(full_train_dataset)
val_size = int(datasets_size * VALIDATION_SPLIT_RATIO)
train_size = datasets_size - val_size

g = torch.Generator().manual_seed(THE_SEED)  # For reproducibility

# Create the actual training and validation datasets using the indices
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_val_dataset, val_indices)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

BATCH_SIZE = 32

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, # We shuffle the training data to prevent the model from learning the order of the images
    num_workers=0, # Use multiple processes for faster data loading
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # No need to shuffle the validation data
    num_workers=0,
)


# Create an instance of the model
model = TBClassifier()


# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # Move the model to the GPU if available

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

#number of epochs
EPOCHS = 10

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS):
    for epoch in range(epochs):
        # Training Loop
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Move data to GPU if available

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs) # Forward pass: get predictions
            loss = criterion(outputs, labels) # Compute the loss
            loss.backward() # Backward pass: compute gradients
            optimizer.step() # Update weights

            running_loss += loss.item()
    
    #Validation Loop
    model.eval() # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad(): # No need to compute gradients during validation
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Epoch {epoch+1}/{epochs}, "
          f"Training Loss: {running_loss/len(train_loader):.4f}, "
          f"Validation Loss: {val_loss/len(val_loader):.4f}, "
          f"Validation Accuracy: {100 * correct / total:.2f}%")
    print("Training complete")

if __name__ == '__main__':
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS)
    # Save the trained model
    torch.save(model.state_dict(), 'tb_classifier.pth')




