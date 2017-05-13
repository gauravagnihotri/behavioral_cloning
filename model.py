'''
Nvidia Arch
using udacity data
'''
# Import packages
import csv
import cv2
import numpy as np
import os

# Load csv file
lines = []
with open(os.getcwd() + '/data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Read images from the sample data  
images = []
measurements = []
for line in lines:
    center_image_path = os.getcwd() + '/data/data/' + line[0] #path for center image
    lft_img = line[1] 
    left_image_path = os.getcwd() + '/data/data/' + lft_img[1:] #path for left image
    rt_img = line[2]
    right_image_path = os.getcwd() + '/data/data/' + rt_img[1:] #path for right image
    center_steering_angle = float(line[3]) #steering angle
    center_image = cv2.imread(center_image_path) 
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    steering_correction_factor = 0.20 # this is a parameter to tune
    left_steering_angle = center_steering_angle + steering_correction_factor
    right_steering_angle = center_steering_angle - steering_correction_factor

    images.extend([center_image, left_image, right_image])
    measurements.extend([center_steering_angle, left_steering_angle, right_steering_angle])

    
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1)) #flip and append all images
    augmented_measurements.append(measurement*-1.0) #flip all steering angles 

X_train = np.array(augmented_images) #create a numpy array for all images
y_train = np.array(augmented_measurements) #create a numpy array for all steering angle measurements

'''
model training and validation
'''    
from keras.models import Sequential
from keras.layers import Lambda
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Cropping2D, Reshape, ELU
from keras.models import Model
import matplotlib.pyplot as plt

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(row, col, ch))) #crop the data to remove sky and trees 
model.add(Lambda(lambda x: x/127.5 - 1.)) # Preprocess incoming data, centered around zero with small standard deviation
model.add(Convolution2D(24,5,5,subsample=(2, 2), border_mode='valid',activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2, 2), border_mode='valid',activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2, 2), border_mode='valid',activation="relu"))
model.add(Convolution2D(64,3,3, border_mode='valid',activation="relu"))
model.add(Convolution2D(64,3,3, border_mode='valid',activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.save('model.h5') #save the model
