# **Behavioral Cloning**


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
---


[//]: # (Image References)

[image4]: ./visualize/flip.png "Image flip"
[image5]: ./visualize/translate.png "Image translate"
[image6]: ./visualize/brightness.png "Image brightness"
[image7]: ./visualize/shadow.png "Image shadow"
[image8]: ./visualize/data.png "Image dataset"


My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### Model architecture

First I tried to use transfer learning with ResNet with provided weights, but on my machine (only CPU training) it was slow to train and unstable, so I decided to switch to simpler model with some adjustments.

Project uses NVIDIA inspired model with additional layer on top to allow model to control color scheme it uses.

Final architecture is:    
* Normalization            
* Cropping layer                 
* Convolution 1x1x3 (Color space choice)
* LeakyReLU activation         
* Convolution 5x5x24             
* LeakyReLU activation          
* Convolution 5x5x36                 
* LeakyReLU activation                
* Convolution 3x3x64                 
* LeakyReLU activation     
* Max pooling 2x2          
* Dropout         
* Convolution 3x3x64                 
* LeakyReLU activation              
* Flatten
* Dense (100)
* LeakyReLU
* Dropout
* Dense (50)
* LeakyReLU
* Dropout
* Dense (10)  
* LeakyReLU
* Dropout
* Dense (1)

To reduce overfitting, the model contains dropout layers. Moreover, different training and validation sets were applied to prevent overfitting on original data, although MSE cannot show real improvement of the learning, so actual performance was measured by running on first and second track.
LeakyReLU has shown itself more effective than ReLU with its capability to restore "dead" neurons.
The model used an adam optimizer with learning rate of 0.001.

#### Training data

To train the model, provided Udacity dataset was used along with captured sections of both tracks.

To capture correct behavior, additionally, I recorded several laps on the first track and second track with some recovery at hard corners and driving on inner radius of the turns to check how it will affect the model behavior. However, later I decided to abandon second track records, as they were representing not suitable data for teaching a model.

Along with center images, I used images from left/right cameras, adjusting the angle accordingly to ±2 degrees.

To augment data set, I used several techniques, including additional shadows, more dark/bright images, flipping and horizontal translation.

![alt text][image7]
![alt text][image6]
![alt text][image4]
![alt text][image5]

After the collection process, I had around 60.000 data points. However, in this data there was strong bias towards 0 steering angle, so I removed 90% of around 0 data and put the data with step of 0.04, which equals 1/25, or 1 degree angle in simulator.

![alt text][image8]
_Final dataset created for training (without augmentation)_

I finally randomly shuffled the data set and put 30% of the data into a validation set. All the augmentations were applied "on the fly" in generator.

I used this data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as later the model started to overfit heavily.

### Network visualizations

[//]: # (Network Image References)

[color_space_0]: ./visualize/layer_color_space_0.png "Color space 0"
[color_space_1]: ./visualize/layer_color_space_1.png "Color space 1"
[color_space_2]: ./visualize/layer_color_space_2.png "Color space 2"
[visual_1_0]: ./visualize/layer_visual_1_0.png "visual_1_0"
[visual_1_1]: ./visualize/layer_visual_1_1.png "visual_1_1"
[visual_1_2]: ./visualize/layer_visual_1_2.png "visual_1_2"
[visual_1_3]: ./visualize/layer_visual_1_3.png "visual_1_3"
[visual_1_4]: ./visualize/layer_visual_1_4.png "visual_1_4"
[visual_1_5]: ./visualize/layer_visual_1_5.png "visual_1_5"
[visual_1_6]: ./visualize/layer_visual_1_6.png "visual_1_6"
[visual_1_7]: ./visualize/layer_visual_1_7.png "visual_1_7"
[visual_1_8]: ./visualize/layer_visual_1_8.png "visual_1_8"
[visual_1_9]: ./visualize/layer_visual_1_9.png "visual_1_9"
[visual_1_10]: ./visualize/layer_visual_1_10.png "visual_1_10"
[visual_1_11]: ./visualize/layer_visual_1_11.png "visual_1_11"
[visual_1_12]: ./visualize/layer_visual_1_12.png "visual_1_12"
[visual_1_13]: ./visualize/layer_visual_1_13.png "visual_1_13"
[visual_1_14]: ./visualize/layer_visual_1_14.png "visual_1_14"
[visual_1_15]: ./visualize/layer_visual_1_15.png "visual_1_15"
[visual_1_16]: ./visualize/layer_visual_1_16.png "visual_1_16"
[visual_1_17]: ./visualize/layer_visual_1_17.png "visual_1_17"
[visual_1_18]: ./visualize/layer_visual_1_18.png "visual_1_18"
[visual_1_19]: ./visualize/layer_visual_1_19.png "visual_1_19"
[visual_1_20]: ./visualize/layer_visual_1_20.png "visual_1_20"
[visual_1_21]: ./visualize/layer_visual_1_21.png "visual_1_21"
[visual_1_22]: ./visualize/layer_visual_1_22.png "visual_1_22"
[visual_1_23]: ./visualize/layer_visual_1_23.png "visual_1_23"

#### Color space choice
![alt text][color_space_0]


![alt text][color_space_1]


![alt text][color_space_2]

#### First layer visualizations
![alt text][visual_1_0]
![alt text][visual_1_1]
![alt text][visual_1_2]
![alt text][visual_1_3]
![alt text][visual_1_4]
![alt text][visual_1_5]
![alt text][visual_1_6]
![alt text][visual_1_7]
![alt text][visual_1_8]
![alt text][visual_1_9]
![alt text][visual_1_10]
![alt text][visual_1_11]
![alt text][visual_1_12]
![alt text][visual_1_13]
![alt text][visual_1_14]
![alt text][visual_1_15]
![alt text][visual_1_16]
![alt text][visual_1_17]
![alt text][visual_1_18]
![alt text][visual_1_19]
![alt text][visual_1_20]
![alt text][visual_1_21]
![alt text][visual_1_22]
![alt text][visual_1_23]
