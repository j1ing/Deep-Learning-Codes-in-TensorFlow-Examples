import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
filepath = "..."

traindf = pd.read_csv(filepath + "train/_annotations.csv")
validdf = pd.read_csv(filepath + "valid/_annotations.csv")

# normalize bounding box coordinates into between 0 and 1
image_width = 416
image_height = 416

xtraindf  = traindf[["xmin","xmax"]]
xvaliddf = validdf[["xmin","xmax"]]

ytraindf  = traindf[["ymin","ymax"]]
yvaliddf = validdf[["ymin","ymax"]]

xtraindf =xtraindf/image_width
xvaliddf = xvaliddf/image_width

ytraindf = ytraindf/image_height
yvaliddf = yvaliddf/image_height

trainlabel = pd.concat([traindf["filename"],xtraindf,ytraindf],axis=1)
validlabel = pd.concat([validdf["filename"],xvaliddf,yvaliddf],axis=1)

label = ["xmin","ymin","xmax","ymax"]

# image generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
imgen = ImageDataGenerator(rescale=1/255)
train = imgen.flow_from_dataframe(trainlabel,filepath+"train/",x_col="filename",y_col=label,target_size=(256,256),class_mode="raw",batch_size=16,shuffle=False)
valid = imgen.flow_from_dataframe(validlabel,filepath+"valid/",x_col="filename",y_col=label,target_size=(256,256),class_mode="raw",batch_size=16,shuffle=False)

# build model
import tensorflow as tf

model = tf.keras.applications.VGG19(input_shape=(256,256,3),include_top=False,weights="imagenet")

model.trainable = False

x = tf.keras.layers.Flatten()(model.output)

x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)

prediction = tf.keras.layers.Dense(4, activation='sigmoid')(x)

model = tf.keras.Model(inputs=model.input, outputs=prediction)

# compile model
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
loss = tf.keras.losses.MeanSquaredError()

model.compile(opt,loss)

rlrop = tf.keras.callbacks.ReduceLROnPlateau("loss",0.7,5,verbose=1)

# training
history = model.fit(train,batch_size=16,epochs=25,validation_data=valid,callbacks=rlrop)

# plot loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()

"""
Epoch 20/25
11/11 [==============================] - 136s 12s/step - loss: 0.0064 - val_loss: 0.0094
Epoch 21/25
11/11 [==============================] - 135s 12s/step - loss: 0.0067 - val_loss: 0.0085
Epoch 22/25
11/11 [==============================] - 141s 13s/step - loss: 0.0085 - val_loss: 0.0077
Epoch 23/25
11/11 [==============================] - 142s 13s/step - loss: 0.0050 - val_loss: 0.0072
Epoch 24/25
11/11 [==============================] - 142s 13s/step - loss: 0.0062 - val_loss: 0.0076
Epoch 00024: ReduceLROnPlateau reducing learning rate to 0.00035000001662410796.
Epoch 25/25
11/11 [==============================] - 142s 13s/step - loss: 0.0055 - val_loss: 0.0076
"""

# load test data
testdf = pd.read_csv(filepath + "test/_annotations.csv")

testdata = imgen.flow_from_dataframe(testdf,filepath+"test/",x_col="filename",target_size=(256,256),batch_size=16,shuffle=False)

# predict test data
y_pred = model.predict(testdata)

# rescale bounding box coordinates based on desired image height and width
y_pred = y_pred*416

y_pred = np.round(y_pred)

# compile predictions into a dataframe with true coordinates
preddf = pd.DataFrame(y_pred)

preddf = preddf.rename(columns={0:"predxmin",1:"predymin",2:"predxmax",3:"predymax"})

testpred = pd.concat([testdf,preddf],axis=1)

testpred["predxmin"] = testpred["predxmin"].astype("Int64")
testpred["predymin"] = testpred["predymin"].astype("Int64")
testpred["predxmax"] = testpred["predxmax"].astype("Int64")
testpred["predymax"] = testpred["predymax"].astype("Int64")

# show image with true bounding box (red), and predicted bounding box (green)
import cv2
for i in range(len(testpred)):
    plt.subplot(5,4,i+1)
    img1 = cv2.imread(filepath + "test/" + testpred["filename"][i])
    img2 = cv2.rectangle(img1,(testpred["xmin"][i],testpred["ymin"][i]),(testpred["xmax"][i],testpred["ymax"][i]),(255,0,0),2)
    img3 = cv2.rectangle(img2,(testpred["predxmin"][i],testpred["predymin"][i]),(testpred["predxmax"][i],testpred["predymax"][i]),(0,255,0),2)
    plt.imshow(img3)
