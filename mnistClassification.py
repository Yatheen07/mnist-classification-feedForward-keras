# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:23:07 2018

@author: kmy07
"""

"""Step 1 : Import keras and mins dataset"""
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

"""Step 2: Understand the dataset Shape"""

from keras.utils import to_categorical
print('Training data shape : ', train_images.shape, train_labels.shape)
print('Testing data shape : ', test_images.shape, test_labels.shape)

# Find the unique numbers from the train labels
classes = np.unique(train_labels) #CHecks for unique labels in the traning labels 
nClasses = len(classes) #length of the output
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes) 

"""Step 3 : Visualize the image """

plt.figure(figsize=[10,5]) # pre set the size of the figure 
# Display the first image in training data
plt.subplot(121) # create a subplot
plt.imshow(train_images[3,:,:], cmap='gray') # display the first image in the train data with all rows and colums in grayscale format
plt.title("Ground Truth : {}".format(train_labels[3])) #display the actual result of the image displayed

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_images[15,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_labels[15]))


"""Step 4: Process the data into a linear numpy array"""
# Change from matrix to array of dimension 28x28 to array of dimention 784
dimData = np.prod(train_images.shape[1:]) # convert the pixels of an image with size 28 x 28 into an single array of size 784.
train_data = train_images.reshape(train_images.shape[0], dimData) #reshape train data
test_data = test_images.reshape(test_images.shape[0], dimData) # reshape test data

"""Step 5: Feature Scaling to accomate the pixel range from 0 to 255"""

# Change to float datatype
train_data = train_data.astype('float32') # this is done because scaling the values will lead to float values.
test_data = test_data.astype('float32')

# Scale the data to lie between 0 to 1
train_data /= 255
test_data /= 255

"""Step 6: Standarise the categories using one-hot encoder"""

# Change the labels from integer to categorical data
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

#Display the change for category label using one-hot encoding
print('Original label 0 : ', train_labels[0])
print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[0])

"""Step 7: Create the Network """ 

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,))) # 1st hidden layer
model.add(Dense(512, activation='relu')) # 2nd hidden layer
model.add(Dense(nClasses, activation='softmax')) #softmax for multi class classification

"""Step 8: Compiling the network """

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

"""Step 9 : Training """

history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1, 
                   validation_data=(test_data, test_labels_one_hot))

"""Step 10 : Model Evaluation """

[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

"""Step 11 : Check for Overfitting """

#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

""" Step 12 : Regularisation """

from keras.layers import Dropout
 
model_reg = Sequential()
model_reg.add(Dense(512, activation='relu', input_shape=(dimData,)))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(512, activation='relu'))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(nClasses, activation='softmax'))

""" Step 13 : Performance after regularisation """

model_reg.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history_reg = model_reg.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1, 
                            validation_data=(test_data, test_labels_one_hot))
 
#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history_reg.history['loss'],'r',linewidth=3.0)
plt.plot(history_reg.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history_reg.history['acc'],'r',linewidth=3.0)
plt.plot(history_reg.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

""" Step 14 : Prediction """

# Predict the most likely class
model_reg.predict_classes(test_data[[15],:])

print(test_data[[15],:])

# Predict the probabilities for each class 
result = model_reg.predict(test_data[[15],:])

""" Step 15 :  Checking for the own image """

import cv2 #import openCv package 

image = cv2.imread(r"E:\Mini Projects\Machine Learning\minst image classification\2.png") # read the image drawn using paint
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # convert it to grayscale

test_image = np.array(image) # convert it into numpy array

print(np.prod(test_image.shape[:])) #linearise the model

result_image = test_image.reshape(1, 784) #reshape the numpy array 

result_image = result_image.astype('float32') #change the data type to float
result_image /= 255 # make the pixel range between 0 and 1
print(result_image)

model_reg.predict_classes(result_image) # prediction

print(model_reg.predict(result_image)) # probablity






