import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# class labels
"""
0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck
"""

# data shape
print("x_train  m = {}, image size = {}".format(x_train.shape[0], x_train.shape[1:]))
print("y_train  m = {}".format(y_train.shape[0]))
print("x_test   m = {}, image size = {}".format(x_test.shape[0], x_test.shape[1:]))
print("y_test   m = {}".format(y_test.shape[0]))

"""
x_train  m = 50000, image size = (32, 32, 3)
y_train  m = 50000
x_test   m = 10000, image size = (32, 32, 3)
y_test   m = 10000
"""

# normalize pixel values
x_train = x_train/255
x_test = x_test/255



# One hot encoding labels
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

print("y_train shape = {}".format(y_train[0].shape))
print("y_test shape = {}".format(y_train[0].shape))
"""
y_train shape = (10,)
y_test shape = (10,)
"""

# image augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
imgen = ImageDataGenerator(vertical_flip=True,
                           horizontal_flip=True,
                           rotation_range=0.25,
                           zoom_range=0.25)

dataset = imgen.flow(x=x_train,y=y_train,batch_size=32)

# training model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size= (3,3), strides=(1,1), activation="relu", padding="same", input_shape=(32,32,3)))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size= (3,3), strides=(1,1), activation="relu", padding="same"))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size= (3,3), strides=(1,1), activation="relu", padding="same"))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size= (3,3), strides=(1,1), activation="relu", padding="same"))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size= (3,3), strides=(1,1), activation="relu", padding="same"))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size= (3,3), strides=(1,1), activation="relu", padding="same"))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=128, activation="relu"))
model.add(tf.keras.layers.Dense(units=10, activation="softmax"))

# Model summary
model.summary()
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_83 (Conv2D)           (None, 32, 32, 32)        896       
_________________________________________________________________
batch_normalization_81 (Batc (None, 32, 32, 32)        128       
_________________________________________________________________
conv2d_84 (Conv2D)           (None, 32, 32, 32)        9248      
_________________________________________________________________
batch_normalization_82 (Batc (None, 32, 32, 32)        128       
_________________________________________________________________
average_pooling2d_59 (Averag (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_85 (Conv2D)           (None, 16, 16, 64)        18496     
_________________________________________________________________
batch_normalization_83 (Batc (None, 16, 16, 64)        256       
_________________________________________________________________
conv2d_86 (Conv2D)           (None, 16, 16, 64)        36928     
_________________________________________________________________
batch_normalization_84 (Batc (None, 16, 16, 64)        256       
_________________________________________________________________
average_pooling2d_60 (Averag (None, 8, 8, 64)          0         
_________________________________________________________________
conv2d_87 (Conv2D)           (None, 8, 8, 128)         73856     
_________________________________________________________________
batch_normalization_85 (Batc (None, 8, 8, 128)         512       
_________________________________________________________________
conv2d_88 (Conv2D)           (None, 8, 8, 128)         147584    
_________________________________________________________________
batch_normalization_86 (Batc (None, 8, 8, 128)         512       
_________________________________________________________________
average_pooling2d_61 (Averag (None, 4, 4, 128)         0         
_________________________________________________________________
flatten_24 (Flatten)         (None, 2048)              0         
_________________________________________________________________
dense_42 (Dense)             (None, 128)               262272    
_________________________________________________________________
dense_43 (Dense)             (None, 10)                1290      
=================================================================
Total params: 552,362
Trainable params: 551,466
Non-trainable params: 896
_________________________________________________________________
"""

# Compile model
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

model.compile(optimizer=opt,loss=loss,metrics=["accuracy"])

# training
history = model.fit(dataset, batch_size=32, epochs=5, validation_data=(x_test,y_test))

# plot loss and accuracy
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2,1,2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()

"""
Epoch 1/5
1563/1563 [==============================] - 212s 135ms/step - loss: 1.7571 - accuracy: 0.3765 - val_loss: 1.4932 - val_accuracy: 0.5116
Epoch 2/5
1563/1563 [==============================] - 211s 135ms/step - loss: 1.1569 - accuracy: 0.5862 - val_loss: 0.9945 - val_accuracy: 0.6576
Epoch 3/5
1563/1563 [==============================] - 214s 137ms/step - loss: 0.9537 - accuracy: 0.6613 - val_loss: 0.9211 - val_accuracy: 0.6778
Epoch 4/5
1563/1563 [==============================] - 225s 144ms/step - loss: 0.8317 - accuracy: 0.7080 - val_loss: 0.8080 - val_accuracy: 0.7218
Epoch 5/5
1563/1563 [==============================] - 219s 140ms/step - loss: 0.7495 - accuracy: 0.7378 - val_loss: 0.7678 - val_accuracy: 0.7400
"""

"""
Notes:
It has a similar effect compare to regularization.
"""
