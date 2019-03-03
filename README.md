# **Behavioral Cloning** 

##  Project Writeup 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./examples/end-to-end-architecture.png "NVIDIA Model Architecture"
[image2]: ./data/IMG/center_2018_12_23_23_52_56_437.jpg "Training example"
[image3]: ./data/IMG/center_2018_12_25_00_08_30_859.jpg "Recovery Image"
[image4]: ./data/IMG/center_2018_12_25_00_08_34_947.jpg "Recovery Image"
[image5]: ./data/IMG/center_2018_12_25_00_08_36_036.jpg "Recovery Image"
[image6]: ./data/IMG/center_2018_12_25_00_06_51_541.jpg "Red and white lane"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted

#### 1. Relevant Files

* model.py containing the script to create and train the model
* model.ipynb using jupyter notebook for creation of model.py
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md here, summarizing the results
* video.mp4 video of the car driving autonomously around the track with my trained model uploaded to it

#### 2. Project Testing
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Training and Validation file

The model.py file contains code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. 

#### 4. Final Video
The "video.mp4" file if proof of how my model performed. The .mp4 file was created using the video.py file provided. As a quick explanation, the command below creates the video:
```sh 
python drive.py model.h5 run1
```
Refer to the README.md in [this repository](https://github.com/udacity/CarND-Behavioral-Cloning-P3) for more info on how video.py works



### Model Architecture and Training Strategy

#### 1. NVIDIA End to End Learning for Self-Driving Cars

My final model is based on NVIDIA's end to end network. Some modifications were made in order to make the model apply better to the purposes of this project and reduce overfitting.
![alt text][image1]

I started with a Keras lambda layer to normalize the data (by dividing it by 255 so that it is between 0 and 1) and mean center it (subtract 0.5 to shift the range to {-0.5:0.5}. I then followed the NVIDIA architecture when it came to convolution depths and filter sizes. Other than that, I added 4 dropout layers and used RELU activations in order to break the linearity of the model. See table below for the model architecture

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image       					| 
| Convolution 5x5     	| 2x2 stride, depth = 24, RELU activated       	|
| Convolution 5x5     	| 2x2 stride, depth = 36, RELU activated       	|
| Dropout				| Dropout factor = 0.3   						|
| Convolution 5x5     	| 2x2 stride, depth = 48, RELU activated       	|
| Convolution 3x3     	| 1x1 stride, depth = 64, RELU activated       	|
| Convolution 3x3     	| 1x1 stride, depth = 64, RELU activated       	|
| Dropout				| Dropout factor = 0.3   						|
| Flatten       		| Flatten layer									|
| Fully connected		| output 100, RELU activated					|
| Dropout				| Dropout factor = 0.3   						|
| Fully connected		| output 50, RELU activated						|
| Dropout				| Dropout factor = 0.3   						|
| Fully connected		| output 10, RELU activated                     |
| Fully connected		| output 1, RELU activated                      |



#### 2. Attempts to reduce overfitting in the model

As mentioned previously, the model contains dropout layers in order to reduce overfitting (model.ipynb file cell 3 lines 20,24,27,29). After some trial and error, a dropout factor of 0.3 worked best. I also split the training data to include a validation set and used a split factor of 0.2 (line 35, cell 3) in order to transform 20% of the data collected to validation data.

The model was trained and validated on shuffled data sets to ensure that the model was not overfitting (cell 3, line 35). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track (refer to video.mp4 to see how my model performed).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.ipynb cell 3 line 34). The Mean Square Error (MSE) function was used as a loss function (cell 3, line 34).

#### 4.  Training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving backwards on the track in order to eliminate left/ right turn bias. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Documentation

#### 1. Solution Design Approach

In order to reach the final architecture, several trial and error instances were taken. I started with the LeNet5 network, but it was yielding inconsistent results as early as the training phase. I played with the hyperparameters a little and changed the numbers of Epochs and got slightly better results. Once I tested the model, the car was driving terribly and was not able to make any hard corners and would also drive off the track frequently. I tried cropping the images as well, but it did not yield better results. 
I then switched to the NVIDIA architecture and my car handled way better on the track, but was still driving off. I noticed that my validation loss was significantly bigger than my training loss, which implied overfitting. So I added the dropout layers, tested those and fine tuned my dropout factor (and 0.3 worked best). At this point, the car was driving fairly well, but was still getting lost at the sharpest two corners. It was also driving off whenever the dashed red and white lane markers were there (see image below).

![alt text][image6]


Since I was able to isolate my problem to these two scenarios only, the easiest approach at that point was to collect more data at these two corners as well as at the red and white lane markings.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

For the final model architecture, please refer to the table above. In the end, I added a fully connected layer with an output size of 1 since that is the dimension of our output: it is a simple float as opposed to a one-hot-encoded array since in this case our output is continuous (infinite range of numbers).


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. I then recorded one backwards lap of center lane driving in order to eliminate any turning bias (if the car mostly turns left for example, which seemed to be the case). Here is an example image of center lane driving:

![alt text][image2]

After the first run of data collection process, I had about 6,000 data points since I was only running my model on centered images. At that point, after I had tried a few preprocessing methods and different network architectures, my model was performing fairly well except at a few specific spots on the track. In addition, I noticed from testing the  model at that point that once the car was off the center of the lane, it did not know how to recover and drive back towards the center. So I decided to record more data of the vehicle recovering from the left side and right sides of the road back to center so that it would learn to do so in case it finds itself close to the lane marking. This type of data was a bit tricky to collect since I was doing my best to start recording while my steering angle was already aimed at the center of the lane. That way, my car would only learn to steer back into the lane and not the other way around. The following 3 images show what a recovery from the far left to the middle of the lane looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

I purposely recorded the second run of data mainly at spots where the lane markings were red and white in order to correct for the weak performance of my model in those areas.
Once I recorded about 5,000 more center camera data points, I tested my model and it functioned perfectly at that point. The car reliably stays between the lane lines and drives smoothly along the entire track. I realize that in other applications collecting more data may not be practical and so further data manipulation and augmentation may be necessary. But in this case, I did not need it.


I finally randomly shuffled the data to prevent the model from achieving undesired methods of learning such as data memorization or overfitting. set and put 20% of the data into a validation set. 

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 since it was mainly yielding decreasing results of the training and validation sets. I used an adam optimizer so that manually training the learning rate wasn't necessary.
