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
The model used an adam optimizer with learning rate of 0.001.

#### Training data

To train the model, provided Udacity dataset was used along with captured sections of both tracks.

To capture correct behavior, firstly, I recorded several laps on the first track and second track with some recovery at hard corners. However, later I decided to abandon second track records, as they were representing behavior not suitable for teaching such a model.

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
[image4]: ./visualize/flip.png "Image flip"
[image5]: ./visualize/translate.png "Image translate"
[image6]: ./visualize/brightness.png "Image brightness"
[image7]: ./visualize/shadow.png "Image shadow"
[image8]: ./visualize/data.png "Image dataset"

#### Color space choice
![alt text][color_space_0]
![alt text][color_space_1]
![alt text][color_space_2]
