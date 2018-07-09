# **Behavioral Cloning**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Environment requirements

* Python=3.5
* Tensorflow=1.8.0
* Keras=2.2.0
* opencv-python=3.4.1

Other libraries are same with the libraries provides in Udacity's [Term 1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit).

## Files Submitted and Development Environment

My project includes the following files:

* `model.ipynb` a Jupyter notebook containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup_report.md` summarizing the results
* `videos/track1.mp4` recording of vehicle driving autonomously one lap around track 1 (max speed 30)
* `videos/track2.mp4` recording of vehicle driving autonomously one lap around track 2 (max speed 25)

Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

The tested maximum speeds for each track are (maximum allowed speed needed to be modified inside the code)

* Track 1: 30
* Track 2: 25

The `model.ipynb` contains all the code I used to

1. Setup development environment
2. Load images
3. Preprocessing images
4. Create a custom generator to fit data in real-time for training and validation
5. Build the network architecture for training and validation
6. Training and validation
7. Fine-tuning the model
8. Save the model

The `videos` folder includes two videos which recorded the vehicle driving autonomously one lap around track 1 and track 2. The video are taken at 50 FPS.

## Data Collection, Augmentation and Preparation

### Data Collection

The data collection process is one the most important part of this project. I collected in total 17368 images for track 1 and 42464 images for track 2.

To make the model more robust to different variations, I collect data in the following methods for each track:

1. Driving normally
2. Driving in reserve order
3. Collect more data in recovery (drive from side to center)
4. Collect more data difficult curves (especially for track 2)

### Augmentation

The simulator actually records not only the image from the center camera, but also records images from the left and right angled cameras. I include both of them into the original datasets and adjusted their labels by correction (left +0.2, right -0.2).

I also flipping the images to augment the dataset (the label is also flipped horizontally). Here is an example before (left) and after (right) flipping:

![alt text][image1]
![alt text][image2]

### Preparation for Training

I created a data generator to read the file names and read batches in training. This avoids reading all the images once into memory and speeds up the training process. 

80% of the data points are randomly selected for training and the remaining 20% are used for validation. At the beginning of each epoch, the data generator will re-shuffle the training set.

The resizing and flipping operation conducted during training.

See Part 3 and Part 4 in the code for more details.

## Model Architecture and Training Strategy

I have tried different architectures including LeNet, ResNet and the architecture from Nvidia [1]. My final architecture is adapted from Nvidia's model.

### Summary of the Final Architecture

The summary of architecture is as follows

|      Layer                           |       Output Shape        |   Parameters & Notes                |
|-------------------------             |---------------------------|------------------------             |
|   Normalization                      |   (None, 66, 200, 3)      |  image/255 -0.5                     |
|   CONV-BN-Activation-Dropout         |   24                      | 5x5 kernel, 2x2 strides, dropout:0.2|
|   CONV-BN-Activation-Dropout         |   36                      | 5x5 kernel, 2x2 strides, dropout:0.2|
|   CONV-BN-Activation-Dropout         |   48                      | 5x5 kernel, 2x2 strides, dropout:0.2|
|   CONV-BN-Activation-Dropout         |   64                      | 3x3 kernel, 1x1 strides, dropout:0.2|
|   CONV-BN-Activation-Dropout         |   64                      | 3x3 kernel, 1x1 strides, dropout:0.2|
|   FC-Dropout                         |   1000                    |     dropout:0.4                     |
|   FC-Dropout                         |   100                     |     dropout:0.4                     |
|   FC-Dropout                         |   50                      |     dropout:0.4                     |

The total number of tunable parameters is **1,390,369**.

Explanation of abbreviations:

* `CONV`: convolutional layer
* `BN`: batch normalization layer
* `Activation`: activation layer
* `FC`: fully-connected layer
* `Dropout`: dropout layer

### Some Explanations on the Model Design

* The normalization layer will resize the image from `(160, 320, 3)` to `(66, 200, 3)` according to Nvidia's paper. This should not change the final prediction performance but significantly reduces the number of parameters in my model.
* The model contains dropout layers in order to reduce overfitting after each convolutional layers and fully-connected layers.
* I choose ELU instead of RELU as activation function because ELU performs slightly better in this problem.

### Model parameter tuning

The model used an adam optimizer. I created a scheduler to decay the learning rate by 0.5 for every 3 epochs. The main training part has 12 epochs in total. The initial learning rate is `1e-3` and it decays to `1.25e-4` after the 9th epoch. The scheduling of learning rate helps the training process significantly.

After finishing the main training process, I ran 5 to 10 epochs on smaller learning rate to fine-tune the model parameters.

See Part 5, Part 6 and Part 7 for more details on model design and training.

## Result

The model can drive the car autonomously on both track 1 (at maximum speed 30) and track 2 (at maximum speed 25).

[//]: # (Image References)

[image1]: ./images/img_original.png "Image before preprocessing"
[image2]: ./images/img_flip.png "Image after flipping"
[image3]: ./images/img_resize.png "Image after resizing"
[image4]: ./images/img_flip_resize.png "Image after flipping and resizing"


## References

[1]: Bojarski, Mariusz, et al. "End to end learning for self-driving cars." arXiv preprint arXiv:1604.07316 (2016).