import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
Kaggle Cats and Dogs Dataset
https://www.kaggle.com/competitions/dogs-vs-cats/data?select=train.zip
"""

# filepath
filepath = "/Users/jeffling/Downloads/dogs-vs-cats/train/"

# list of directory
import os
trainlist = os.listdir(filepath)

"""
['dog.8011.jpg',
 'cat.5077.jpg',
 'dog.7322.jpg',
 'cat.2718.jpg',
 'cat.10151.jpg',
 ...]
"""

# data size
len(trainlist)
"""
25000
"""

# create a training dataframe with filenames and labels
df = pd.DataFrame(trainlist,columns=["image_id"])
"""
            image_id
0       dog.8011.jpg
1       cat.5077.jpg
2       dog.7322.jpg
"""

df["label"] = 0

"""
           image_id  label
0       dog.8011.jpg      0
1       cat.5077.jpg      0
2       dog.7322.jpg      0
"""

from tqdm import tqdm
for i in tqdm(range(len(df))):
    label = df["image_id"][i].split(".")[0]
    df["label"][i] = label

"""
           image_id label
0       dog.8011.jpg   dog
1       cat.5077.jpg   cat
2       dog.7322.jpg   dog
3       cat.2718.jpg   cat
"""

# split the data into training set and validation set
from sklearn.model_selection import train_test_split
train, test = train_test_split(df,test_size=0.2,random_state=101)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
imgen = ImageDataGenerator(rescale=1/255,
                           horizontal_flip=True,
                           vertical_flip=True,
                           zoom_range=0.2,
                           rotation_range=0.2)

train_data = imgen.flow_from_dataframe(dataframe=train,
                                       directory=filepath,
                                       x_col="image_id",
                                       y_col="label",
                                       target_size=(128,128),
                                       class_mode="binary",
                                       seed=101,
                                       batch_size=16)

validation_data = imgen.flow_from_dataframe(dataframe=test,
                                            directory=filepath,
                                            x_col="image_id",
                                            y_col="label",
                                            target_size=(128,128),
                                            class_mode="binary",
                                            seed=101,
                                            batch_size=16)

# class indices
train_data.class_indices
"""
{'cat': 0, 'dog': 1}
"""

# Download VGG19 pre-trained model and weights
model = tf.keras.applications.VGG19(include_top=False, # set to False to customize the output layer according to the number of class
                                    weights="imagenet",
                                    input_shape=(128,128,3))

model.trainable = False

x = tf.keras.layers.Flatten()(model.output)
x = tf.keras.layers.Dense(8, activation='relu')(x)
prediction = tf.keras.layers.Dense(1, activation='sigmoid')(x) # 1 unit, binary classification, sigmoid

#Creating model object
model = tf.keras.Model(inputs=model.input, outputs=prediction)

model.summary()
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_17 (InputLayer)        [(None, 128, 128, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 128, 128, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 128, 128, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 64, 64, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 64, 64, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 64, 64, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 32, 32, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 32, 32, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 32, 32, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 32, 32, 256)       590080    
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 32, 32, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 16, 16, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 16, 16, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 16, 16, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 16, 16, 512)       2359808   
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 16, 16, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 8, 8, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 8, 8, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 8, 8, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 8, 8, 512)         2359808   
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 8, 8, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
_________________________________________________________________
flatten_36 (Flatten)         (None, 8192)              0         
_________________________________________________________________
dense_64 (Dense)             (None, 8)                 65544     
_________________________________________________________________
dense_65 (Dense)             (None, 1)                 9         
=================================================================
Total params: 20,089,937
Trainable params: 65,553
Non-trainable params: 20,024,384
_________________________________________________________________
"""

# compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

model.compile(optimizer=opt,loss=loss,metrics=["accuracy"])

# training
history = model.fit(train_data,batch_size=16,epochs=5,validation_data=validation_data)

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
1250/1250 [==============================] - 4684s 4s/step - loss: 0.5128 - accuracy: 0.7440 - val_loss: 0.3933 - val_accuracy: 0.8192
Epoch 2/5
1250/1250 [==============================] - 4780s 4s/step - loss: 0.3972 - accuracy: 0.8144 - val_loss: 0.3784 - val_accuracy: 0.8280
Epoch 3/5
1250/1250 [==============================] - 4770s 4s/step - loss: 0.3812 - accuracy: 0.8236 - val_loss: 0.3625 - val_accuracy: 0.8402
Epoch 4/5
1250/1250 [==============================] - 4779s 4s/step - loss: 0.3644 - accuracy: 0.8332 - val_loss: 0.3698 - val_accuracy: 0.8346
Epoch 5/5
1250/1250 [==============================] - 4753s 4s/step - loss: 0.3665 - accuracy: 0.8331 - val_loss: 0.3582 - val_accuracy: 0.8390
"""