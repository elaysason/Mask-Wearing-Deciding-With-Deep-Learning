import os
import argparse
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import torch.nn as nn
from skimage import io


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))


        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        

        self.fc = nn.Linear(8 * 8 * 128, 2)
        self.dropout = nn.Dropout(p=0.8)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return self.logsoftmax(out)

class SampleDataset(Dataset):
    def __init__(self, dir, transforms):
        self.dir = dir
        self.transforms = transforms
        self.images_names = os.listdir(self.dir)

        self.images = self.create_entries()

    def create_entries(self):
        data = []

        for image_name in self.images_names:
            data.append({'x': image_name.split('_')[0], 'y': int(image_name.split('_')[1][0])})
        return data

    def __getitem__(self, index):
        entry = self.images[index]
        image = io.imread(os.path.join(self.dir, (entry['x'] + '_' + str(entry['y']) + '.jpg')))
        image = Image.fromarray(np.uint8(image))
        image_new = self.transforms(image)

        return image_new, torch.tensor(entry['y']), self.images_names[index]

    def __len__(self):
        return len(self.images)

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder

#####

model = torch.load('model.pkl')
data_transform =data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), transforms.RandomChoice([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), transforms.ColorJitter(brightness=0.7,
                                                                    contrast=0.7,
                                                                    saturation=0.7)
        ]), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

image_dataset = SampleDataset(args.input_folder, data_transform)

dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=5, shuffle=True)
data = {'prediction':[],'name':[]}
for (images, labels, names) in dataloader:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    data['prediction'].extend(predicted.cpu().detach().numpy())
    data['name'].extend(names)
predicted_df = pd.DataFrame(data)


# TODO - How to export prediction results
predicted_df.to_csv("prediction.csv", index=False, header=False)
