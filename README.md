
## Ear Biometrics Detection
In this project, I have done earlocalisation, and used severa; Steps and matched ear by using algorithms such as sift skin detection and determined whether the ear has been matched or not.

## YCbCr images
YCbCr (sometimes written as Y'CbCr) is a color space used in digital imaging systems, particularly in video and image compression. It represents images using three components: Y (luminance or brightness), Cb (chrominance blue), and Cr (chrominance red). Here's a brief overview of each component:

Y (luminance): This represents the brightness information of the image. It's essentially a grayscale version of the image.
Cb (chrominance blue) and Cr (chrominance red): These components represent the color information in the image. They represent the difference between the luminance (Y) and the blue and red components of the color, respectively. These components are used to represent color information while reducing redundancy, as the human eye is more sensitive to changes in brightness than changes in color.
YCbCr is widely used in digital video and image compression standards like JPEG, MPEG, and H.264 because it separates the luminance and chrominance components, allowing for more efficient compression. By compressing the chrominance components more heavily than the luminance component, it's possible to reduce the file size while maintaining good image quality, as the human eye is less sensitive to changes in color detail compared to changes in brightness detail.

When you see YCbCr images, they typically have three separate channels or planes representing Y, Cb, and Cr values. These values are usually represented as numbers in a certain range, with Y typically ranging from 0 to 255 and Cb and Cr often ranging from -128 to 127 or 0 to 255 depending on the specific implementation.

Converting an image from RGB (Red, Green, Blue) color space to YCbCr involves complex mathematical transformations, but it's a standard process implemented in most image processing software. The reverse process, converting from YCbCr back to RGB, is also possible.







## Morphological operations

Morphological operations are a fundamental set of techniques used in image processing and computer vision for analyzing and processing images based on shapes. These operations are primarily used for tasks like noise reduction, edge detection, image enhancement, and segmentation.

Here are some common morphological operations:

Erosion: Erosion is used to shrink the boundaries of foreground objects in an image. It works by moving a structuring element (a small matrix or kernel) over the image and replacing each pixel with the minimum pixel value within the neighborhood defined by the structuring element. Erosion is useful for removing small objects and fine details from an image.
Dilation: Dilation is the opposite of erosion. It expands the boundaries of foreground objects in an image. Like erosion, it also uses a structuring element, but it replaces each pixel with the maximum pixel value within the neighborhood defined by the structuring element. Dilation is helpful for filling in small holes in objects and enlarging features in an image.
Opening: Opening is a combination of erosion followed by dilation. It's useful for removing small objects and smoothing the boundaries of larger objects while preserving the overall shape of the objects. Opening is often used for noise reduction and removing small artifacts.
Closing: Closing is the reverse of opening. It's a combination of dilation followed by erosion. Closing is useful for filling in small gaps between objects and closing small breaks in object boundaries. It's commonly used for tasks like connecting broken lines and filling holes in objects.
Morphological Gradient: The morphological gradient is the difference between the dilation and erosion of an image. It highlights the edges of objects in the image and is useful for edge detection and segmentation tasks.
Top Hat and Bottom Hat: Top hat and bottom hat operations are used to enhance specific features in an image. The top hat operation is the difference between the input image and its opening, while the bottom hat operation (also known as black hat) is the difference between the closing and the input image. These operations are useful for enhancing small details and structures in an image.

## SIFT Detection

SIFT (Scale-Invariant Feature Transform) is a powerful algorithm for extracting distinctive features from images, which are invariant to scale, rotation, and illumination changes. These features can then be used for various computer vision tasks such as object recognition, image stitching, and 3D reconstruction. Here's a basic overview of how SIFT works for feature extraction:

Scale-space Extrema Detection: SIFT detects keypoints (interest points) in an image by identifying local extrema in the scale-space representation of the image. This involves convolving the image with Gaussian filters at multiple scales to create a pyramid of blurred images. Keypoints are detected at locations where the difference of Gaussian (DoG) function across scales reaches local maxima or minima.
Keypoint Localization: Once potential keypoints are detected, SIFT refines their locations to sub-pixel accuracy by fitting a quadratic function to the DoG function to estimate the keypoint's position. It also discards keypoints with low contrast or those located on edges, which are less distinctive.
Orientation Assignment: SIFT assigns an orientation to each keypoint based on the local gradient directions around the keypoint. This ensures that the descriptors computed for each keypoint are rotationally invariant. Typically, histograms of gradient orientations are computed in a neighborhood around the keypoint, and the dominant orientation is selected as the keypoint's orientation.
Descriptor Generation: After keypoint localization and orientation assignment, SIFT computes a descriptor for each keypoint to describe its local appearance. The descriptor is a vector representation that captures information about the gradient magnitudes and orientations in a local neighborhood around the keypoint. The descriptor is designed to be invariant to changes in scale, rotation, and illumination.
Descriptor Matching: Once descriptors are computed for keypoints in multiple images, feature matching is performed to establish correspondences between keypoints in different images. This is typically done by comparing the distance between descriptors using techniques like Euclidean distance or cosine similarity. Robust matching techniques are used to filter out incorrect matches and find reliable correspondence

## CNN
CNN stands for Convolutional Neural Network, which is a type of artificial neural network commonly used in image recognition and classification tasks, although they can be applied to other types of data as well. CNNs have revolutionized the field of computer vision and have achieved remarkable success in various tasks such as object detection, image segmentation, and facial recognition.

Here's a basic overview of how CNNs work:

Convolutional Layers: CNNs consist of multiple layers, and the first few layers are typically convolutional layers. In these layers, the network applies a set of learnable filters (also known as kernels) to the input image. Each filter extracts certain features from the input image by performing a convolution operation, which involves sliding the filter over the input image and computing dot products at each position. The result is a set of feature maps that capture different aspects of the input image.
Activation Function: After each convolutional operation, a non-linear activation function is applied element-wise to the output of the convolutional layer. The most commonly used activation function in CNNs is the Rectified Linear Unit (ReLU), which introduces non-linearity into the network and helps the network learn complex patterns in the data.
Pooling Layers: Pooling layers are used to reduce the spatial dimensions of the feature maps while retaining important information. The most common type of pooling operation is max pooling, where the maximum value within a certain neighborhood (e.g., a 2x2 window) is retained, and the rest are discarded. Pooling helps in reducing computational complexity, controlling overfitting, and achieving translation invariance.
Fully Connected Layers: After several convolutional and pooling layers, the high-level features learned from the input image are flattened into a vector and fed into one or more fully connected (dense) layers. These layers perform classification based on the learned features. The final layer usually employs a softmax activation function to output class probabilities.
Training: CNNs are trained using a variant of the backpropagation algorithm called stochastic gradient descent (SGD) or its variants (e.g., Adam, RMSprop). During training, the network learns to minimize a loss function, which measures the difference between the predicted output and the ground truth labels. The weights of the network are adjusted iteratively to minimize this loss function using gradient descent.
Regularization: To prevent overfitting, various regularization techniques such as dropout, L2 regularization, and data augmentation are commonly employed in CNNs. Dropout randomly deactivates some neurons during training to prevent co-adaptation of features, while L2 regularization penalizes large weight values to prevent overfitting.

## KNN
KNN stands for K-Nearest Neighbors, which is a simple yet effective algorithm used for both classification and regression tasks in machine learning. It's a type of instance-based learning where the model doesn't explicitly learn a function from the training data but instead memorizes the training dataset and makes predictions based on similarity measures between new data points and the training instances.

Here's how the KNN algorithm works:

Training: KNN doesn't involve explicit training in the same way as many other machine learning algorithms. During the "training" phase, the algorithm simply stores the feature vectors and corresponding labels of the training data.
Prediction:
For classification: Given a new, unseen data point, the algorithm calculates the distance (typically Euclidean distance) between the new data point and every point in the training dataset.
For regression: Instead of voting for the majority class, KNN takes the average of the labels of the K-nearest neighbors to predict the output for the new data point.
Choosing K: The "K" in KNN refers to the number of nearest neighbors to consider when making a prediction. This is a hyperparameter that needs to be chosen prior to making predictions. The choice of K can significantly affect the performance of the algorithm. Smaller values of K can lead to more complex decision boundaries, while larger values of K can lead to smoother decision boundaries.
Majority Voting (for classification): Once the K nearest neighbors are identified, in the case of classification, the algorithm assigns the class label that is most common among the K neighbors to the new data point. This is known as majority voting.
Weighted Voting: Optionally, you can use weighted voting, where the contribution of each neighbor to the prediction is weighted by its distance from the new data point. Closer neighbors have a greater influence on the prediction than farther ones.

## Algorithms implemented





![image](https://github.com/Prayag-Chawla/Ear-biometrics-Detection/assets/92213377/1d5fad4b-d5d3-464f-b7dd-3c6b1e175fdf)

![image](https://github.com/Prayag-Chawla/Ear-biometrics-Detection/assets/92213377/59e21609-ecf5-4fe7-9c34-5a2e032fae0f)

![image](https://github.com/Prayag-Chawla/Ear-biometrics-Detection/assets/92213377/e51882cc-13a2-4ebb-a380-582246b62aa6)
![image](https://github.com/Prayag-Chawla/Ear-biometrics-Detection/assets/92213377/336f061c-b0b5-453e-8842-e787b6f199aa)
![image](https://github.com/Prayag-Chawla/Ear-biometrics-Detection/assets/92213377/fbe1fc0a-acd9-44bf-99af-059e0b1bbee5)
![image](https://github.com/Prayag-Chawla/Ear-biometrics-Detection/assets/92213377/9be1c0e7-c968-4688-a13c-6ea864bfe31e)


## Model used
The primary model which has been used is KNN and CNN.

## Libraries and Usage

```
#IMPORTING ALL THE LIBRARIES

import os
import cv2
import numpy as np
```






## Accuracy
There was a very high Accuracy from the model as we were able to get the decision variables from requiredvarious models





## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```
## ACCURACY
The model has an accuracy of 0.8809523809523809

## Used By
In the real world, this project is used in the biometrics industry extensivelyy.
## Appendix

A very crucial project in the realm of data science and new age biometrics domain using visualization techniques as well as machine learning modeling.

## Tech Stack

**Client:** Python, Naive byes classifier, gaussian naive byes, support vector machine, random forest, decision tree classifier, logistic regression model, EDA analysis, machine learning, sequential model of ML, SHAP explainer model, data visualization libraries of python.


## OUTPUT
![image](https://github.com/Prayag-Chawla/Ear-biometrics-Detection/assets/92213377/1a5f7c58-4763-446e-921c-10c9d5cd4043)
![image](https://github.com/Prayag-Chawla/Ear-biometrics-Detection/assets/92213377/d40fccb4-9e00-4a20-8e24-5a9e16598b40)

![image](https://github.com/Prayag-Chawla/Ear-biometrics-Detection/assets/92213377/f4fa1420-c7e8-4c06-bd5e-89bcf83f2e42)
![image](https://github.com/Prayag-Chawla/Ear-biometrics-Detection/assets/92213377/3c8c27a7-d8a7-4734-b3ad-8866e755c9d1)


## Feedback

If you have any feedback, please reach out to us at chawlapc.619@gmail.com

