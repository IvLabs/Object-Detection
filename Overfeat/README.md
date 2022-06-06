# RACCOON DETECTOR

The objective of this task is to perform object detection using the sliding window algorithm and further implementing overfeat research paper for improved results.

[Overfeat research paper link](https://arxiv.org/abs/1312.6229)

Dataset link
https://github.com/datitran/raccoon_dataset
## DESCRIPTION:
In Sliding window algorithm, a window of fixed size which slides over the entire image and is one of the classical technique for object detection.

## IMPLEMENTATION
First, the dataset is created. It consists of over 200 images.The neural netwrok gives us the feature map which then fed to the classifier and regressor heads separately.Both the heads are trained separately. 

After their successful training, the sliding window technique is implemented on high resoultion images. 
In sliding window detection, a fixed window size is passed over the image and convolute the part of the image under the window and predict the output (class_of_object,bounding_box_coordinates).Then the sliding window is slided on next part of image and the process is repeated over entire image and with different window sizes.
But this method is computationally expensive and time consuming. Hence, we use the convolutional implemenatation of the sliding window algorithm where all the fully connected layers are replaced with convolutional layers.

We apply convolutional techniques to the whole image and output a feature map in which each cell corresponds to a  stride of sliding window(as shown in the image below). The output of each cell is whether the object is present or not in that sliding window. In our case we trained the model on raccoon dataset and hence our model only predicts presence of raccoon in the image. 
![](https://i.imgur.com/MjvbmOs.png)

It is a two-stage object detector in which we get the feaure maps and then they are fed to classifier and regressor separately.

Using ideas from the overfeat research paper, the tasks of localization and classification are optimized. For this we train a CNN model on image classifier and then we replace the top classifier layers by a regression network and train it to predict object bounding boxes at each spatial location and scale.

## RESULTS : 
![](https://i.imgur.com/IMNHc83.png)
![](https://i.imgur.com/7bH9f2Y.png)
![](https://i.imgur.com/J5f1el4.png)
![](https://i.imgur.com/6fcXcmk.png)
![](https://i.imgur.com/EdjtnVT.png)

LOSS vs no of iterations :

![](https://i.imgur.com/9t2ruSE.png)

## DEPENDENCIES :
* Opencv
* Pandas
* Matplotlib
* PyTorch
* NumPy
* PIL
