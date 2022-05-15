import torch
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage import io
from sklearn import metrics
from sklearn.metrics import f1_score
from torch.utils.data.dataset import Dataset
from torchvision import transforms

warnings.filterwarnings('ignore')

num_epochs = 30
batch_size = 35
learning_rate = 0.001
image_size = 100

## Note: The part of getting the graphs of our result we put in comment in order the file could compile on the vm

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


def evluate_hw1(test_loader):
    model = torch.load('model.pkl')
    cur_f1 = []
    cur_loss = []
    all_outputs_test = []
    all_labels_test = []
    for (images, labels, names) in test_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        all_outputs_test.extend(torch.exp(outputs).cpu().detach().numpy()[:, 1])
        all_labels_test.extend(labels.cpu().detach().numpy())
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        cur_loss.append(loss.cpu().detach().numpy())
        cur_f1.append(f1_score(labels.cpu(), predicted.cpu()))
    epoch_loss = sum(cur_loss) / len(cur_loss)
    epoch_f1 = sum(cur_f1) / len(cur_f1)
    return epoch_loss, epoch_f1, all_labels_test, all_outputs_test


##def imshow(inp, title=None):
##    """Imshow for Tensor."""
##    inp = inp.numpy().transpose((1, 2, 0))
##    mean = np.array([0.485, 0.456, 0.406])
##    std = np.array([0.229, 0.224, 0.225])
##    inp = std * inp + mean
##    inp = np.clip(inp, 0, 1)
##    plt.imshow(inp)
##    if title is not None:
##        plt.title(title)
##    plt.pause(0.001)  # pause a bit so that plots are updated


##def roc_graph(all_labels, all_outputs, set):
##    fpr, tpr, _ = metrics.roc_curve(all_labels, all_outputs)
##
##    # create ROC curve
##    plt.plot(fpr, tpr, label=set)
##    plt.legend()
##    plt.title('ROC AUC graph')
##    plt.ylabel('True Positive Rate')
##    plt.xlabel('False Positive Rate')

def write_to_file(data,key):
  with open(key+'.txt', 'w') as filehandle:
    for listitem in data[key]:
        filehandle.write('%s\n' % listitem)
if __name__ == "__main__":
    data_transforms = {'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), transforms.RandomChoice([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), transforms.ColorJitter(brightness=0.7,
                                                                    contrast=0.7,
                                                                    saturation=0.7)
        ]), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
        'test': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(), transforms.RandomChoice([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(), transforms.ColorJitter(brightness=0.4,
                                                                        contrast=0.4,
                                                                        saturation=0.4)
            ]), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_dir = ''
    image_datasets = {x: SampleDataset(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                   for x in ['train', 'test']}

    cnn = CNN()

    if torch.cuda.is_available():
        cnn = cnn.cuda()

    # convert all the weights tensors to cuda()
    # Loss and Optimizer

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    data = {}
    data['epoch_loss_train'] = []
    data['epoch_f1_train']  = []
    data['epoch_loss_test'] = []
    data['epoch_f1_test'] = []
    data['all_outputs_train'] = []
    data['all_outputs_test'] = []
    data['all_labels_train'] = []
    data['all_labels_test'] = []
    


    for epoch in range(num_epochs):
        all_outputs_train = []
        all_labels_train = []
        cur_loss = []
        cur_f1 = []
        i = 0
        for i, (images, labels, names) in enumerate(dataloaders['train']):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            outputs = cnn(images)
            data['all_outputs_train'].extend(torch.exp(outputs).cpu().detach().numpy()[:, 1])
            data['all_labels_train'].extend(labels.cpu().detach().numpy())
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            cur_loss.append(loss.cpu().detach().numpy())
            cur_f1.append(f1_score(labels.cpu(), predicted.cpu()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1,
                         len(image_datasets['train']) // batch_size, loss.data))
            i += 1

        torch.save(cnn, 'model.pkl')
        test_loss, test_f1, data['all_labels_test'], data['all_outputs_test'] = evluate_hw1(dataloaders['test'])
        data['epoch_loss_test'].append(test_loss)
        data['epoch_f1_test'].append(test_f1)
        data['epoch_loss_train'].append(sum(cur_loss) / len(cur_loss))
        data['epoch_f1_train'].append(sum(cur_f1) / len(cur_f1))
        for key in data.keys():
          write_to_file(data,key)
## roc_graph(data['all_labels_train'], data['all_outputs_train'], 'Train')
## roc_graph(data['all_labels_test'], data['all_outputs_test'], 'Test')
## plt.show()
## plt.plot(np.arange(num_epochs), data['epoch_loss_train'], label="Train")
## plt.plot(np.arange(num_epochs), data['epoch_loss_test'], label="Test")
## plt.xlabel("epoch")
## plt.ylabel("loss")
## plt.legend()
## plt.title("Compering test and train loss")
## plt.show()
## plt.plot(np.arange(num_epochs), data['epoch_f1_train'], label="Train")
## plt.plot(np.arange(num_epochs), data['epoch_f1_test'], label="Test")
## plt.xlabel("epoch")
## plt.ylabel("f1 score")
## plt.legend()
## plt.title("Compering test and train f1 score")
## plt.show()
