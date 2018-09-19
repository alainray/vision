## Computer Vision Knowledge Repository

### Topic list



#### Machine Learning Jargon

* Backbone:
* Convolutional Neural Network
* 

#### Computer Vision Jargon
* Action:
* Actor:

* Background: everything that is not an object in an image.
* Classification: This image includes a dog.
* Localization: The dog is here.
* Semantic Segmentation: All the highlighted things are dogs.
* Instance Segmentation: This is the location of each and every dog.
* Bounding Box: a 4-tuple of real values denoting height, width and the coordinates of the left uppermost point of a box that encloses an instance of a class.
* Mask: the evolution of the bounding box taken to a pixel level.
* Streams: probably refers to both RGB and Flow input data in video datasets.


#### Arch.itectures
* AlexNet
* VGG-16:
* ResNet 50/101:
* R-CNN
* Fast R-CNN
* SPPNet
* Region Proposal Network (RPN)
* Faster R-CNN
* Feature Pyramid Network (FPN)
* Mask R-CNN
* Fully Convolutional Neural Network (FCN)

#### Loss Functions
* Cross Entropy Loss:
* Smooth L2? (from Fast R-CNN paper):
#### Accuracy measures
* mAP
* IoU: Intersection over Union. Basically, how much two rectangles overlap. Usually used to determine how close a proposed bounding box is to a ground truth box. Can be used in masks as well, however, I do not know how to calculate it then.
* Average Recall (AR): ??
* AUC: Area under Curve. Area under the Average Recall - Average Precision curve.
#### Papers
