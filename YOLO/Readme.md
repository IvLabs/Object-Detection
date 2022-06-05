# YOLO v1
## Description
Object detection using YOLO v1 algorithm using PyTorch over Pascal VOC dataset.
YOLO is an one-stage detector, instead of passing the feature maps to separate classifier and regressor heads obtained from the backbone of the network, it passes entire image at once and predicts the bounding box coordinates, confidences score and class probabilites.

## Implementation
* Pascal VOC 2012 dataset is used for training which consists of 20 classes
* The labels are modulated into the the tensor of size (Batchsize,S,S,N*5+20)
* Pretrained Resnet18 model is used as backbone and additional fully connected layers are added to produce desired output.
```
Bounding box = [xc,yc,w,h,c]
```
xc and yc are centre of bounding box parameterized to be offset of that particular grid cell.
w and h are width and height wrt. whole image.
c is the confidence score

* In loss function, it penalizes the bounding boxand classes probability only if an object is present in the grid cell.
* Class confidence as Pr(object)*IOU 
* Model is trained over 40 epoch with different learning rates deacy of 10^-1 after every 10 epochs.
* The newlayers are also finetuned using the test dataset, which then achieves accuracy of 70.30% over Pascal VOC test set
## Results
> training loss

> mAP over test dataset

## Dependencies
 PyTorch
 NumPy
 Torchvision
 Albumentations 1.1.0
 OpenCV
 opencv-python-headless 4.5.2.52
