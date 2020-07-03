##################################################
# Imports
##################################################

import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model

#write path to the folder where data are stored
os.chdir(r'C:\Users\Vittorino\Google Drive\Data Science - Unipd\I Year\II Semester\Algorithmic Methods and Machine Learning\Machine Learning\Project2\data')
currentDirectory=os.getcwd()#current working directory


##################################################
# Params
##################################################
DATA_BASE_FOLDER = currentDirectory

##################################################
# Load dataset
##################################################
x_train = np.load(os.path.join(DATA_BASE_FOLDER, 'train.npy'))
x_valid = np.load(os.path.join(DATA_BASE_FOLDER, 'validation.npy'))
x_test = np.load(os.path.join(DATA_BASE_FOLDER, 'test.npy'))
y_train = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'train.csv'))['class'].values
y_valid = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'validation.csv'))['class'].values
y_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Plot random images of different classes
plt.figure(figsize=(25, 5))
for idx in range(20):
    plt.subplot(1, 20, idx + 1)
    img = x_train[idx].reshape(28, 28)
    plt.title(f'{y_labels[y_train[idx]]}')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.show()


##################################################
# Process the data here, if needed
##################################################
x_train=x_train/255
x_valid=x_valid/255
trial=x_train.reshape(50000,28,28,1)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[(Y.reshape(-1)-1)]
    return Y

y_train = convert_to_one_hot(y_train, 10)
y_valid = convert_to_one_hot(y_valid, 10)

##################################################
# Implement you model here
##################################################
def model1(input_shape):
    X_input = Input(input_shape)
    X = Conv2D(filters=16,
               kernel_size=(5, 5), 
               strides=(2, 2), 
               name='conv0', 
               padding='same',
               activation='relu')(X_input)#'same' preserve the dimension of input through the convolutions
    X= MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(X)
    X=Conv2D(filters=16, kernel_size=(5, 5), strides=(2,2), padding='same', activation='relu')(X)
    X=MaxPooling2D((2, 2), strides=(2, 2), padding='same')(X)
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(units=10, activation=None)(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='model1')

    return model
  
input_shape=(28,28,1)  
model1 = model1(input_shape)
model1.summary()


model1.compile(optimizer = "Adam", 
               loss =  'categorical_crossentropy',
               metrics = ["accuracy"])


model1.fit(x = trial, y = y_train, epochs = 4, batch_size = 25)


##################################################
# Evaluate the model here
##################################################

# Use this function to evaluate your model
def accuracy(y_pred, y_true):
    '''
    input y_pred: ndarray of shape (N,)
    input y_true: ndarray of shape (N,)
    '''
    return (1.0 * (y_pred == y_true)).mean()

# Report the accuracy in the train and validation sets.