# SEMANTIC SEGMENTATION
## (U-NET ARCHITECTURE)

The task is to perform Semantic Segmentation on images using U-NET architecture and PASCAL-VOC 2007 dataset to train the model.
 
## DESCRIPTION
Semantic Segmentation is a process of partitioning a digital image into multiple image segments, known as image objects. It is being applied in various fields like autonomous driving, robotic navigation, localization, and scene understanding.

In this process every pixel in the image is assigned a particular label. Our model is trained on the PASCAL-VOC 2007 dataset and hence it can classify objects in an image into 21 different types of classes (including background class) and performs semantic segmentation on them.

## IMPLEMENTATION
The model uses the U-NET architecture to perform semantic segmentation on a given input image.
The U-net architecture  consists of a contracting path (left side) and an expansive path (right side)(as shown in the image).
![](https://i.imgur.com/zUS6LCw.png)

* In the network a cascade of convolution layers and a pooling layer is applied to the input image and down sampling/encoding is performed. This process of applying convoltional layers and downsampling is repeated and we reach the bottleneck layer.This path is known as the contracting path.
* From the bottleneck again a cascade of up-convolution and pooling layers is applied and up sampling is performed until in the final layer 1x1 convolutions are used to map each component feature vector to the desired number of classes. This path is known as the expansive path. 
* A concatenation with the correspondingly cropped feature map from the contracting path is done at each step before applying the up-convolutional layers
* In total 23 convolutional layers are used.
* In training the model is trained using stochastic gradient descent(High momentum=0.99)
* The loss is calculated as the cross entropy loss.

## LIMITATIONS 
Semantic segmentation treats multiple objects of the same class as a single entity.
Means the model can not differentiate between two different objects of the same class (for example : If there are two humans in an image, both the humans would be classified as person and not like person1 and person2. 
 
## DEPENDENCIES :
* Albumentations 1.1.0
* OpenCV-python-headless version == 4.5.2.52
* PyTorch
* NumPy
* PIL
* Matplotlib

## RESULTS
![](https://i.imgur.com/FY7kQXB.jpg)


![](https://i.imgur.com/qhCwU68.jpg)
![](https://i.imgur.com/GdewydC.jpg)
![](https://i.imgur.com/0hfzMPC.jpg)
![](https://i.imgur.com/TIeNWV8.jpg)

![](https://i.imgur.com/FLJpVrH.png)

![](https://i.imgur.com/L2S9aHE.png)
