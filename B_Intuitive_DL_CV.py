"""
Marc D. Holman
CIS 2532 - Advanced Python Programming / Intro. Data Science
5 / 4 / 2020

Module 11 Deep Learning

Program B:  Intuitive Deep Learning, Convolutional Neural Networks
for Computer Vision

This program was taken from Github repository:
https://github.com/josephlee94/intuitive-deep-learning
comments are based on Joseph Lee's

Output is attached as a separate text file:  'Output_for_B.txt'
and plots include following image files:

"""

from keras.datasets import cifar10
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

#  download the dataset which includes,
#  images to be recognized
#  labels - 10 possible labels airplane automobile bird cat dog deer frog horse ship and truck
#  size 60000 images, 50000 training, 10000 testing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#  shape of our data set
print('x_train shape:', x_train.shape)
print('y_train.shape: ', y_train.shape)

#  take a look at an individual image, first image of training dataset
print(x_train[0])

#  to see the actual image
img = plt.imshow(x_train[0])
plt.savefig('frog.png')
plt.show()
print('The label is:', y_train[0])

#  look at another image, index 1 in our training set
img = plt.imshow(x_train[1])
plt.savefig('truck.png')
plt.show()
print('The label is:', y_train[1])

#  we want the probabilioty of each of the 10 different classes.
#  convert to one hot encoding
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)
print('The one hot label is:', y_train_one_hot[1])

#  Let the values be between 0 and 1 to aid in training the network.  Since pixel values
#  already take the values between 0 and 255 we simply need to divide by 255
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

print(x_train[0])


###### BUILDING AND TRAINING CONVOLUTIONAL NEURAL NETWORK ####

#  first define the architecture
model = Sequential()

#  first layer in code looks like:
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32,32,3)))

# second layer
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

# max pooling later with pool size 2 x 2 and stride 2
model.add(MaxPooling2D(pool_size=(2, 2)))

#  dropout layer with probability of 0.25 to prevent overfitting
model.add(Dropout(0.25))

# four more similar layers except the depth of the conv layer is 46
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#  code in the fully connected layer
model.add(Flatten())

# add a dense layer of 512 neurons with relu activation
model.add(Dense(512, activation='relu'))

#  we add another dropout of probability 0.5:
model.add(Dropout(0.5))

#  dense (FC) layer with 10 neurons and soft max activation
model.add(Dense(10, activation='softmax'))

#  output a full summary of the full architecture
print(model.summary())

#  code the loss function using Adam, a type of stochastic descent
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#  finally, run our training with a batch size 32 and 20 epochs
hist = model.fit(x_train, y_train_one_hot,
           batch_size=32, epochs=20,
           validation_split=0.2)

#  training is done, now we can visualize the model and validation loss
#  as well as training / validation accuracy over the number of epochs
#  using below code:
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('model_lossB.png')
plt.show()


#  visualize model accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.savefig('model_accuracyB.png')
plt.show()


#####  TEST OUT WITH YOUR OWN IMAGES ####
model.evaluate(x_test, y_test_one_hot)[1]

#   save the trained model - will be saved in a file format called HDF5 (.h5)
model.save('my_cifar10_model.h5')