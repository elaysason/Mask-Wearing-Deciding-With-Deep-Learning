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

### More-About-The-Task-In-Hand
Wearing mask in this is not neccarly inuative. A person is wearing a mask if and only if 3 or more parts of his face are covered defined by this picture

<img src="https://i.imgur.com/LIojmcB.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width=50% height=50% />
## Program Structure
The first part is specific functions for the dataset. The second part is the implementaions of the algorithms:
* model.py - Creation of the network and evlautes its performance and visualize on the test set. 
* predict.py - Uses the model created to, given a path to new pictures outputs a csv file with the prections on those images.

### Network Structure

## Installation
I will use google as an example, but similar process can be performed on other notebook editors
1. Open google Colab
2. Clone the project by:
	```
	!git clone https://github.com/elaysason/ANALYSIS-SP500.git
	```
    <img src="https://i.imgur.com/IYmNxac.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width=40% height=40% />
3. Now the folder is in your files on colab. Simpily download the notebook as showed

    <img src="https://i.imgur.com/BIY19HC.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width=30% height=30% />
