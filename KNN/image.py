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
os.chdir(r'C:\Users\Giuli\Google Drive\Data Science - Unipd\I Year\II Semester\Algorithmic Methods and Machine Learning\Machine Learning\Project2\data')
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
x_train_reshape=x_train.reshape(50000,28,28,1)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[(Y.reshape(-1)-1)]
    return Y

y_train_reshape = convert_to_one_hot(y_train, 10)
y_valid_reshape = convert_to_one_hot(y_valid, 10)

##################################################
# Implement you model here
##################################################

##### Neural Network ####
def model1(input_shape):
    X_input = Input(input_shape)
    X = Conv2D(filters=32,
               kernel_size=(3, 3), 
               strides=(1, 1), 
               name='conv0', 
               padding='same',
               activation='relu')(X_input)#'same' preserve the dimension of input through the convolutions
    X= MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(X)
    X=Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(X)
    X=MaxPooling2D((2, 2), strides=(2, 2), padding='same')(X)
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(units=10, activation='softmax')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='model1')

    return model
  
input_shape=(28,28,1)  
model1 = model1(input_shape)
model1.summary()


model1.compile(optimizer = "adam", 
               loss =  'categorical_crossentropy',
               metrics = ["accuracy"])


model1.fit(x = x_train_reshape, y = y_train_reshape, epochs = 4, batch_size = 25)

x_valid_reshape=x_valid.reshape(10000,28,28,1)
preds=model1.evaluate(x=x_valid_reshape,y=y_valid_reshape)
print()
print ("Loss = " + str(preds[0]))
print ("Valid set Accuracy = " + str(preds[1]))


##################################################
# Implement you model here
##################################################

import joblib

###### KNN #####
#scaling the data (StandardScaler & MinMaxScaler tested) is maybe not really necessary
#since all feature have the same unit of measurement (int in [0;255]).
#Trying with and without standardization, the train and valid set perfomance is
#the same.
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = StandardScaler()#MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_train=x_train_scaled
x_valid=x_valid_scaled

#For better numerical stability is advisible to divide by 255(if not already done in preprocessing)
x_train=x_train/255
x_valid=x_valid/255



#***************************************************************************
#to run only if you want to work with less data to speed up computation in experimental phase
x_train=x_train[0:1000]
y_train=y_train[0:1000]
x_valid=x_valid[0:1000]
y_valid=y_valid[0:1000]
#*******************************************************************************

#use grid search to find the best value of k
from sklearn.neighbors import KNeighborsClassifier
param_grid = {'n_neighbors': list(range(1,21))}
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
grid_search = GridSearchCV(KNeighborsClassifier(p=1), param_grid, cv=cv,n_jobs=-1)
grid_search.fit(x_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Valid set score: {:.2f}".format(grid_search.score(x_valid, y_valid)))

# save the model to disk
filename = 'grid_search.sav'
joblib.dump(grid_search, filename)

# to load the model from disk
grid_search = joblib.load(filename)

results = pd.DataFrame(grid_search.cv_results_)
results
k_star=grid_search.best_params_['n_neighbors']

#------------------------------------------------
#maybe useless given the grid search used above
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 21)
for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, p=1)
    clf.fit(x_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(x_train, y_train.reshape(-1,1)))
    # record generalization accuracy
    test_accuracy.append(clf.score(x_valid, y_valid.reshape(-1,1)))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
#---------------------------------------------------


#to recognize an item, we don't look at the single pixel, but at the overall image.
#So extracting PCA and using them to identify different items is an attempt to mimic
#how humans identify items.
#Using the optimal value of k found before, we look how much varies the accuracy
#in train and valid set increasing the n° of PC used to approximate each sample.
#Around 20 PCs, the Accuracy curve starts being flat, meaning that adding further 
#PCs increases just a little the performance, so no worth to add them.
from sklearn.decomposition import PCA
#training_accuracy = []
test_accuracy = []
PC_number = range(1, 101)
for i in PC_number:
    pca = PCA(n_components=i, whiten=True, random_state=0).fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_valid_pca = pca.transform(x_valid)
    #print("x_train_pca.shape: {}".format(x_train_pca.shape))
    knn = KNeighborsClassifier(n_neighbors=k_star,p=1)
    knn.fit(x_train_pca, y_train)
    # record training set accuracy
    #training_accuracy.append(knn.score(x_train_pca, y_train[0:1000].reshape(-1,1)))
    # record generalization accuracy
    test_accuracy.append(knn.score(x_valid_pca, y_valid.reshape(-1,1)))
#plt.plot(PC_number, training_accuracy, label="training accuracy")
plt.plot(PC_number, test_accuracy, label="valid accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n° PC")
plt.legend()  
plt.savefig('Accuracy VS number_PCs.png')

#optimal number of PC
PCs_star=np.argmax(test_accuracy)+1
#observing the plot, the accuracy growth becomes flat approx. after the 
#first 20 PCs. So to have an easier model, with less feature, we 
#decide to use 20 PCs.
PCs_star=20
valid_set_accuracy=test_accuracy[PCs_star-1]

#train the model with k=k_star and PCs_star=20
pca = PCA(n_components=20, whiten=True, random_state=0).fit(x_train)
x_train_pca = pca.transform(x_train)
x_valid_pca = pca.transform(x_valid)

# save the model to disk
filename = 'pca.sav'
joblib.dump(pca, filename)
# to load the model from disk
pca = joblib.load(filename)

knn = KNeighborsClassifier(n_neighbors=k_star,p=1)
knn.fit(x_train_pca, y_train)
valid_set_accuracy_20=knn.score(x_valid_pca, y_valid.reshape(-1,1))
#this is the same model with 20 PCs trained and tested in above for cycle,
#so valid_set_accuracy == valid_set_accuracy_20

# save the model to disk
filename = '5nn.sav'
joblib.dump(knn, filename)
# to load the model from disk
knn = joblib.load(filename)

#plot the PCs
print("pca.components_.shape: {}".format(pca.components_.shape))
fix, axes = plt.subplots(4, 5, figsize=(15, 12),subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape((28,28)),cmap='viridis')
    ax.set_title("{}. component".format((i + 1)))
plt.savefig('first 20 PCs.png')




##################################################
# Evaluate the model here
##################################################

# Use this function to evaluate your model
y_pred=knn.predict(x_valid_pca)
def accuracy(y_pred, y_true):
    '''
    input y_pred: ndarray of shape (N,)
    input y_true: ndarray of shape (N,)
    '''
    return (1.0 * (y_pred == y_true)).mean()

# Report the accuracy in the train and validation sets.
valid_set_accuracy_20_helper=accuracy(y_pred,y_valid)

