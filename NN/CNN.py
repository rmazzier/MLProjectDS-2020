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
    X = Conv2D(filters=16,
               kernel_size=(5, 5), 
               strides=(1, 1), 
               name='conv0', 
               padding='same',
               activation='relu')(X_input)#'same' preserve the dimension of input through the convolutions
    X= MaxPooling2D(pool_size=(4, 4), strides=(1, 1), padding='same')(X)
    X=Conv2D(filters=16, kernel_size=(5, 5), strides=(1,1), padding='same', activation='relu')(X)
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


model1.fit(x = x_train_reshape, y = y_train, epochs = 4, batch_size = 25)

x_valid_reshape=x_valid.reshape(10000,28,28,1)
preds=model1.evaluate(x=x_valid_reshape,y=y_valid)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

