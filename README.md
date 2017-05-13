# Behaviorial Cloning Project

Overview
---
This project uses a modified version of [Nvidia's Architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) to train a model to clone driving behavior. The model is built using Keras.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

* [Linux Simulator Link](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
### Model Architecture and Training Strategy
#### 1. Using Single Layer Model

I tried a single layer model to verify if everything is working. The single layer model input is shown as follows
```
model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(row, col, ch)))
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Flatten())
model.add(Dense(1))
model.summary()
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.save('model.h5') #save the model
```

#### 1. Single Layer Model Output

```
Using TensorFlow backend.
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 65, 320, 3)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 62400)         0           lambda_1[0][0]                   
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1)             62401       flatten_1[0][0]                  
====================================================================================================
Total params: 62,401
Trainable params: 62,401
Non-trainable params: 0
____________________________________________________________________________________________________
Train on 38572 samples, validate on 9644 samples
Epoch 1/2
38572/38572 [==============================] - 24s - loss: 1.1026 - acc: 0.0734 - val_loss: 4.1483 - val_acc: 0.0362
Epoch 2/2
38572/38572 [==============================] - 23s - loss: 1.5817 - acc: 0.0633 - val_loss: 2.0330 - val_acc: 0.0555
```
The single layer model has high validation loss and very small validation accuracy. The car doesn't stay on road while in autonomous mode. 
But this architecture helps in confirming that all prerequisites are met. 

#### 2. Using LENET Architecture 
The next architecture used was LENET, since LENET is a very first convolutional architecture developed to recognize characters 

![LENET Architecture](http://www.pyimagesearch.com/wp-content/uploads/2016/06/lenet_architecture.png)

Fig. shows the flow of LENET Arch [1]

```
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(row, col, ch)))
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.summary()
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1)
model.save('model_3.h5')
```
#### 2. LENET Model Output
```
Using TensorFlow backend.
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 65, 320, 3)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 61, 316, 6)    456         lambda_1[0][0]                   
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 30, 158, 6)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 26, 154, 6)    906         maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 13, 77, 6)     0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 6006)          0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 120)           720840      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 84)            10164       dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             85          dense_2[0][0]                    
====================================================================================================
Total params: 732,451
Trainable params: 732,451
Non-trainable params: 0
____________________________________________________________________________________________________
Train on 38572 samples, validate on 9644 samples
Epoch 1/1
38572/38572 [==============================] - 25s - loss: 0.0563 - acc: 0.1790 - val_loss: 0.0223 - val_acc: 0.1835
```

LENET architecture is complex enough to train the car to go round half of the track, however, the car tries to correct itself too often, resulting in not very smooth performance. The validation loss kept increasing with number of epochs, hence only one epoch was used. The vehicle also drives closer to the edge of the track rather than the center. The vehicle could complete the lap without getting off the road, however the performance is not very consistent and in a separate run, the vehicle brushed with the edge of the bridge. 

[![Lenet Architecture Implementation](https://i.ytimg.com/vi/gLNZs3Dik_U/1.jpg)](https://youtu.be/gLNZs3Dik_U)

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 


## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

## References 

[1] http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
