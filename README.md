# Mask-Wearing-Deciding-With-Deep-Learning

Detrimane if a person wears a mask or not using deep learning. 
1. [General](#General)
    - [Background](#background)
    - [More About The Task In Hand](#More-About-The-Task-In-Hand)
3. [Program Structure](#Program-Structure)
    - [Network Structure](#Network-Structure)
5. [Installation](#Installation)
6. [Footnote](#footnote)
## General
The goal is to use deep learning in general and convolutional neural network in particular convolutional neural networks in order to detranimine if the person in the picture wears a mask or not.

### Background
CNN is kind of the deep neural network, used mostly to analyze images. The goal of CNN is to reduce the dimensions of the picture to make analyze and prediction easier without losing important features. It uses filters, each one of them will represnt a certian features. In the start the features will be simple such as horizontal or diagonal edges, afterwards the filters will repsent more complex features such as wheels or eyes.

<img src="https://i.imgur.com/rXyYqjI.png" width = 70% height=70%>

### More-About-The-Task-In-Hand
Wearing mask in this is not neccarly inuative. A person is wearing a mask if and only if 3 or more parts of his face are covered defined by this picture

<img src="https://i.imgur.com/LIojmcB.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width=40% height=40% />

## Program Structure
The first part is specific functions for the dataset. The second part is the implementaions of the algorithms:
* model.py - Creation of the network and evlautes its performance and visualize on the test set. 
* predict.py - Uses the model created to, given a path to new pictures outputs a csv file with the prections on those images.

### Network-Structure
The model is a 3 layers CNN that includes: 

The first layer:
Number of in channels is 3 because the items are RGB and the number of out channels is 16
The batches are being normalized to 16.
Then, it uses ReLu as an activation function and maxpooling the result.

The second layer:
Number of in channels is 16(as the first layer's output out channels) and the number of out channels is 24.
The batches are being normalized to 24.
Then, it uses ReLu as an activation function and maxpooling the result.

The third layer:
Number of in channels is 24 (as the first layer's output out channels) and the number of out channels is 32.
The batches are being normalized to 32.
Then, it uses ReLu as an activation function and maxpooling the result.

the model drops out 70% for regularization, in order to prevent overfitting.

The fc layer - The fully connected layer is a layer where each input is connected to all neurons. It helps provide and optimize the classification.

Finally, the model computes log soft max.


## Installation
1. Open the terminal

2. Clone the project by:
```
    $ git clone https://github.com/elaysason/Mask-Wearing-Deciding-With-Deep-Learning.git
```
3. Run the predict.py file by:
```
    $ python predict.py /path/to/folder
```
## Footnote
The folder used as an input from predcit.py should hold images in the following format XXXXXX_Y.jpg:  XXXXXX is an identificator for the image and Y is the label 0 for no mask and 1 for mask.
