"""
Marc D. Holman
CIS 2532 - Advanced Python Programming / Intro. Data Science
5 / 4 / 2020

Module 11 Deep Learning

Program A:  Build your first Neural Network to predict housing prices with Keras

This program was taken from Github repository:
https://github.com/josephlee94/intuitive-deep-learning

Output is attached as a separate text file:  'Output_for_A.txt'
and plots include:
'Model3_loss_accuracy.png'
'model_loss.png'
'overfitting_accuracy.png'
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras import regularizers

#  load the csv data into a pandas DataFrame
df = pd.read_csv('housepricedata.csv')

#  take a look at our data
print("\nTake a Look at the Data: \n")
print(df.head())

#  convert the dataframe to an array by accessing it's values attribute
dataset = df.values

print("\nDataFrame converted to an array: \n")
print(dataset)

#  split the dataset into input features / the labels we wish to predict
X = dataset[:, 0:10]
Y = dataset[:, 10]

#  normalize the data
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

print("\nTake a look at X_scale: \n")
print(X_scale)

#  use the train test split function to divide our dataset into training and testing data
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print("\nTake a look at the shape of our data: \n")
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

#  we are using the sequential model - we need to describe the layers in sequence
#  This network has 3 layers, 2 Hidden layers and 1 output layer
model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

#  before training begins it's necessary to configure the model.  We tell it:
#  what algorithm to use for optimization - stochastic descent
#  what loss function to use - binary cross entropy
model.compile(optimizer='sgd',
               loss='binary_crossentropy',
              metrics=['accuracy'])

#  training only requires a single line of code, a function called fit, which
#  fits the parameters to the data.  We specify:
#  what data we are training
#  the size of our mini batch
# how long we want to train it for in epochs
# validation data
hist = model.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))

#  finally, we evaluate our data on the test set
print(model.evaluate(X_test, Y_test)[1])


#### VISUALIZING LOSS AND ACCURACY ###########

#  visualize the training and validation loss:
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig("model_loss.png")
plt.show()

#  alternative visualization, model accuracy
# plt.plot(hist.history['acc'])    -------------->  This line throws an error in PyCharm, but not in jupyter notebooks
# plt.plot(hist.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='lower right')
# plt.savefig('model_accuracy.png')
# plt.show()


###  Adding regularization to the network ###

#  train a model which will overfit
model_2 = Sequential([
    Dense(1000, activation='relu', input_shape=(10,)),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1, activation='sigmoid'),
])
model_2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist_2 = model_2.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))

#  another visualization for overfitting
plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig("overfitting_accuracy.png")
plt.show()

#  we see some overfitting in Model 2, incoporate L2 regularization and dropout in 3rd model
model_3 = Sequential([
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(10,)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
])

model_3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist_3 = model_3.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))

#  plot loss and accuracy graphs for model 3
plt.plot(hist_3.history['loss'])
plt.plot(hist_3.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.ylim(top=1.2, bottom=0)
plt.savefig("Model3_loss_accuracy.png")
plt.show()


