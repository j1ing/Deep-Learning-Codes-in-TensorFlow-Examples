import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
Kaggle Cats and Dogs Dataset
https://www.kaggle.com/competitions/dogs-vs-cats/data?select=train.zip
"""

# filepath
filepath = "......"

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

# Download MobileNet pre-trained model and weights
model = tf.keras.applications.mobilenet.MobileNet(include_top=False, # set to False to customize the output layer according to the number of class
                                                    weights="imagenet",
                                                  alpha = 1,
                                                  pooling="avg",
                                                    input_shape=(128,128,3))
"""
If alpha < 1.0, proportionally decreases the number of filters in each layer.
If alpha > 1.0, proportionally increases the number of filters in each layer.
If alpha = 1.0, default number of filters from the paper are used at each layer.
"""
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
input_21 (InputLayer)        [(None, 128, 128, 3)]     0         
_________________________________________________________________
conv1 (Conv2D)               (None, 64, 64, 32)        864       
_________________________________________________________________
conv1_bn (BatchNormalization (None, 64, 64, 32)        128       
_________________________________________________________________
conv1_relu (ReLU)            (None, 64, 64, 32)        0         
_________________________________________________________________
conv_dw_1 (DepthwiseConv2D)  (None, 64, 64, 32)        288       
_________________________________________________________________
conv_dw_1_bn (BatchNormaliza (None, 64, 64, 32)        128       
_________________________________________________________________
conv_dw_1_relu (ReLU)        (None, 64, 64, 32)        0         
_________________________________________________________________
conv_pw_1 (Conv2D)           (None, 64, 64, 64)        2048      
_________________________________________________________________
conv_pw_1_bn (BatchNormaliza (None, 64, 64, 64)        256       
_________________________________________________________________
conv_pw_1_relu (ReLU)        (None, 64, 64, 64)        0         
_________________________________________________________________
conv_pad_2 (ZeroPadding2D)   (None, 65, 65, 64)        0         
_________________________________________________________________
conv_dw_2 (DepthwiseConv2D)  (None, 32, 32, 64)        576       
_________________________________________________________________
conv_dw_2_bn (BatchNormaliza (None, 32, 32, 64)        256       
_________________________________________________________________
conv_dw_2_relu (ReLU)        (None, 32, 32, 64)        0         
_________________________________________________________________
conv_pw_2 (Conv2D)           (None, 32, 32, 128)       8192      
_________________________________________________________________
conv_pw_2_bn (BatchNormaliza (None, 32, 32, 128)       512       
_________________________________________________________________
conv_pw_2_relu (ReLU)        (None, 32, 32, 128)       0         
_________________________________________________________________
conv_dw_3 (DepthwiseConv2D)  (None, 32, 32, 128)       1152      
_________________________________________________________________
conv_dw_3_bn (BatchNormaliza (None, 32, 32, 128)       512       
_________________________________________________________________
conv_dw_3_relu (ReLU)        (None, 32, 32, 128)       0         
_________________________________________________________________
conv_pw_3 (Conv2D)           (None, 32, 32, 128)       16384     
_________________________________________________________________
conv_pw_3_bn (BatchNormaliza (None, 32, 32, 128)       512       
_________________________________________________________________
conv_pw_3_relu (ReLU)        (None, 32, 32, 128)       0         
_________________________________________________________________
conv_pad_4 (ZeroPadding2D)   (None, 33, 33, 128)       0         
_________________________________________________________________
conv_dw_4 (DepthwiseConv2D)  (None, 16, 16, 128)       1152      
_________________________________________________________________
conv_dw_4_bn (BatchNormaliza (None, 16, 16, 128)       512       
_________________________________________________________________
conv_dw_4_relu (ReLU)        (None, 16, 16, 128)       0         
_________________________________________________________________
conv_pw_4 (Conv2D)           (None, 16, 16, 256)       32768     
_________________________________________________________________
conv_pw_4_bn (BatchNormaliza (None, 16, 16, 256)       1024      
_________________________________________________________________
conv_pw_4_relu (ReLU)        (None, 16, 16, 256)       0         
_________________________________________________________________
conv_dw_5 (DepthwiseConv2D)  (None, 16, 16, 256)       2304      
_________________________________________________________________
conv_dw_5_bn (BatchNormaliza (None, 16, 16, 256)       1024      
_________________________________________________________________
conv_dw_5_relu (ReLU)        (None, 16, 16, 256)       0         
_________________________________________________________________
conv_pw_5 (Conv2D)           (None, 16, 16, 256)       65536     
_________________________________________________________________
conv_pw_5_bn (BatchNormaliza (None, 16, 16, 256)       1024      
_________________________________________________________________
conv_pw_5_relu (ReLU)        (None, 16, 16, 256)       0         
_________________________________________________________________
conv_pad_6 (ZeroPadding2D)   (None, 17, 17, 256)       0         
_________________________________________________________________
conv_dw_6 (DepthwiseConv2D)  (None, 8, 8, 256)         2304      
_________________________________________________________________
conv_dw_6_bn (BatchNormaliza (None, 8, 8, 256)         1024      
_________________________________________________________________
conv_dw_6_relu (ReLU)        (None, 8, 8, 256)         0         
_________________________________________________________________
conv_pw_6 (Conv2D)           (None, 8, 8, 512)         131072    
_________________________________________________________________
conv_pw_6_bn (BatchNormaliza (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_pw_6_relu (ReLU)        (None, 8, 8, 512)         0         
_________________________________________________________________
conv_dw_7 (DepthwiseConv2D)  (None, 8, 8, 512)         4608      
_________________________________________________________________
conv_dw_7_bn (BatchNormaliza (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_dw_7_relu (ReLU)        (None, 8, 8, 512)         0         
_________________________________________________________________
conv_pw_7 (Conv2D)           (None, 8, 8, 512)         262144    
_________________________________________________________________
conv_pw_7_bn (BatchNormaliza (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_pw_7_relu (ReLU)        (None, 8, 8, 512)         0         
_________________________________________________________________
conv_dw_8 (DepthwiseConv2D)  (None, 8, 8, 512)         4608      
_________________________________________________________________
conv_dw_8_bn (BatchNormaliza (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_dw_8_relu (ReLU)        (None, 8, 8, 512)         0         
_________________________________________________________________
conv_pw_8 (Conv2D)           (None, 8, 8, 512)         262144    
_________________________________________________________________
conv_pw_8_bn (BatchNormaliza (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_pw_8_relu (ReLU)        (None, 8, 8, 512)         0         
_________________________________________________________________
conv_dw_9 (DepthwiseConv2D)  (None, 8, 8, 512)         4608      
_________________________________________________________________
conv_dw_9_bn (BatchNormaliza (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_dw_9_relu (ReLU)        (None, 8, 8, 512)         0         
_________________________________________________________________
conv_pw_9 (Conv2D)           (None, 8, 8, 512)         262144    
_________________________________________________________________
conv_pw_9_bn (BatchNormaliza (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_pw_9_relu (ReLU)        (None, 8, 8, 512)         0         
_________________________________________________________________
conv_dw_10 (DepthwiseConv2D) (None, 8, 8, 512)         4608      
_________________________________________________________________
conv_dw_10_bn (BatchNormaliz (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_dw_10_relu (ReLU)       (None, 8, 8, 512)         0         
_________________________________________________________________
conv_pw_10 (Conv2D)          (None, 8, 8, 512)         262144    
_________________________________________________________________
conv_pw_10_bn (BatchNormaliz (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_pw_10_relu (ReLU)       (None, 8, 8, 512)         0         
_________________________________________________________________
conv_dw_11 (DepthwiseConv2D) (None, 8, 8, 512)         4608      
_________________________________________________________________
conv_dw_11_bn (BatchNormaliz (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_dw_11_relu (ReLU)       (None, 8, 8, 512)         0         
_________________________________________________________________
conv_pw_11 (Conv2D)          (None, 8, 8, 512)         262144    
_________________________________________________________________
conv_pw_11_bn (BatchNormaliz (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_pw_11_relu (ReLU)       (None, 8, 8, 512)         0         
_________________________________________________________________
conv_pad_12 (ZeroPadding2D)  (None, 9, 9, 512)         0         
_________________________________________________________________
conv_dw_12 (DepthwiseConv2D) (None, 4, 4, 512)         4608      
_________________________________________________________________
conv_dw_12_bn (BatchNormaliz (None, 4, 4, 512)         2048      
_________________________________________________________________
conv_dw_12_relu (ReLU)       (None, 4, 4, 512)         0         
_________________________________________________________________
conv_pw_12 (Conv2D)          (None, 4, 4, 1024)        524288    
_________________________________________________________________
conv_pw_12_bn (BatchNormaliz (None, 4, 4, 1024)        4096      
_________________________________________________________________
conv_pw_12_relu (ReLU)       (None, 4, 4, 1024)        0         
_________________________________________________________________
conv_dw_13 (DepthwiseConv2D) (None, 4, 4, 1024)        9216      
_________________________________________________________________
conv_dw_13_bn (BatchNormaliz (None, 4, 4, 1024)        4096      
_________________________________________________________________
conv_dw_13_relu (ReLU)       (None, 4, 4, 1024)        0         
_________________________________________________________________
conv_pw_13 (Conv2D)          (None, 4, 4, 1024)        1048576   
_________________________________________________________________
conv_pw_13_bn (BatchNormaliz (None, 4, 4, 1024)        4096      
_________________________________________________________________
conv_pw_13_relu (ReLU)       (None, 4, 4, 1024)        0         
_________________________________________________________________
global_average_pooling2d (Gl (None, 1024)              0         
_________________________________________________________________
flatten_37 (Flatten)         (None, 1024)              0         
_________________________________________________________________
dense_66 (Dense)             (None, 8)                 8200      
_________________________________________________________________
dense_67 (Dense)             (None, 1)                 9         
=================================================================
Total params: 3,237,073
Trainable params: 8,209
Non-trainable params: 3,228,864
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
1250/1250 [==============================] - 227s 181ms/step - loss: 0.2737 - accuracy: 0.8740 - val_loss: 0.1648 - val_accuracy: 0.9320
Epoch 2/5
1250/1250 [==============================] - 240s 192ms/step - loss: 0.1696 - accuracy: 0.9308 - val_loss: 0.1659 - val_accuracy: 0.9324
Epoch 3/5
1250/1250 [==============================] - 241s 193ms/step - loss: 0.1541 - accuracy: 0.9358 - val_loss: 0.1627 - val_accuracy: 0.9322
Epoch 4/5
1250/1250 [==============================] - 269s 215ms/step - loss: 0.1467 - accuracy: 0.9392 - val_loss: 0.1926 - val_accuracy: 0.9202
Epoch 5/5
1250/1250 [==============================] - 289s 231ms/step - loss: 0.1498 - accuracy: 0.9368 - val_loss: 0.1481 - val_accuracy: 0.9406
"""
