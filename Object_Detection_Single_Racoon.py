import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Racoon dataset can be downloaded from Roboflow

# load data
filepath = ".../Raccoon/"
traindf = pd.read_csv(filepath + "train/_annotations.csv")
validdf = pd.read_csv(filepath + "valid/_annotations.csv")

# locate the center of the object
traindf["xc"] = 0
traindf["yc"] = 0
for i in tqdm(range(len(traindf))):
    x = (traindf["xmax"][i]+traindf["xmin"][i])/2
    y = (traindf["ymax"][i]+traindf["ymin"][i])/2
    traindf["xc"][i] = x
    traindf["yc"][i] = y

validdf["xc"] = 0
validdf["yc"] = 0
for i in tqdm(range(len(validdf))):
    x = (validdf["xmax"][i]+validdf["xmin"][i])/2
    y = (validdf["ymax"][i]+validdf["ymin"][i])/2
    validdf["xc"][i] = x
    validdf["yc"][i] = y

# examine the center x,y through plotting the image
import cv2
m=1
img = cv2.imread(filepath + "train/"+traindf["filename"][m])
s = traindf[traindf["filename"]==traindf["filename"][m]]
s = s.reset_index()
for i in tqdm(range(len(s))):
    bx = np.int(s["xc"][i])
    by = np.int(s["yc"][i])
    cv2.circle(img,(bx,by),5,(0,0,255),-1)
    x1 = np.int(s["xmin"][i])
    y1 = np.int(s["ymin"][i])
    x2 = np.int(s["xmax"][i])
    y2 = np.int(s["ymax"][i])
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
plt.imshow(img)

# normalize the coordinates
w = traindf["width"]
h = traindf["height"]
traindf["xmin"] = traindf["xmin"]/w
traindf["ymin"] = traindf["ymin"]/h
traindf["xmax"] = traindf["xmax"]/w
traindf["ymax"] = traindf["ymax"]/h
traindf["xc"] = traindf["xc"]/w
traindf["yc"] = traindf["yc"]/h

w = validdf["width"]
h = validdf["height"]
validdf["xmin"] = validdf["xmin"]/w
validdf["ymin"] = validdf["ymin"]/h
validdf["xmax"] = validdf["xmax"]/w
validdf["ymax"] = validdf["ymax"]/h
validdf["xc"] = validdf["xc"]/w
validdf["yc"] = validdf["yc"]/h


# spplit the image into 4X4 grid
grid = 4

m=50
img = cv2.imread(filepath + "train/"+traindf["filename"][m])
img = cv2.resize(img,(400,400))
for i in range(grid):
    cv2.line(img,(int(400/grid)*i,0),(int(400/grid)*i,400),(0,255,0),1)
    cv2.line(img,(0,int(400/grid)*i),(400,int(400/grid)*i),(0,255,0),1)
s = traindf[traindf["filename"]==traindf["filename"][m]]
s = s.reset_index()
for i in tqdm(range(len(s))):
    xc = np.int(s["xc"][i]*400)
    yc = np.int(s["yc"][i]*400)
    cv2.circle(img,(xc,yc),5,(0,0,255),-1)
    x1 = np.int(s["xmin"][i]*400)
    y1 = np.int(s["ymin"][i]*400)
    x2 = np.int(s["xmax"][i]*400)
    y2 = np.int(s["ymax"][i]*400)
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
plt.imshow(img)

# assign grid numbers to each center x,y
grid = 4

traindf["gridx"] = np.floor(traindf["xc"]*grid)
traindf["gridy"] = np.floor(traindf["yc"]*grid)

traindf["gridnum"] = traindf["gridx"]+traindf["gridy"]*grid

# the position of the center x,y are relative the the grid cells they were assigned to
# bx,by
traindf["bx"] = (traindf["xc"]-(traindf["gridx"]*int(400/grid)/400))/(int(400/grid)/400)
traindf["by"] = (traindf["yc"]-(traindf["gridy"]*int(400/grid)/400))/(int(400/grid)/400)

# width and height of the bounding boxes are relative to the image as a whole
# bw, bh
traindf["bw"] = traindf["xmax"]-traindf["xmin"]
traindf["bh"] = traindf["ymax"]-traindf["ymin"]


grid = 4

validdf["gridx"] = np.floor(validdf["xc"]*grid)
validdf["gridy"] = np.floor(validdf["yc"]*grid)

validdf["gridnum"] = validdf["gridx"]+validdf["gridy"]*grid

# bx,by
validdf["bx"] = (validdf["xc"]-(validdf["gridx"]*int(400/grid)/400))/(int(400/grid)/400)
validdf["by"] = (validdf["yc"]-(validdf["gridy"]*int(400/grid)/400))/(int(400/grid)/400)

# bw, bh
validdf["bw"] = validdf["xmax"]-validdf["xmin"]
validdf["bh"] = validdf["ymax"]-validdf["ymin"]


# to avoid duplicate images due to multiple objects appearing in the same photo
trainimglist = traindf["filename"].unique().tolist()
validimglist = validdf["filename"].unique().tolist()

X_train = []
t=0
for i in tqdm(trainimglist):
    img = cv2.imread(filepath+"train/"+i)
    img = cv2.resize(img,(400,400))
    img = img/255
    X_train.append(img)
X_train = np.array(X_train)

X_valid = []
t=0
for i in tqdm(validimglist):
    img = cv2.imread(filepath+"valid/"+i)
    img = cv2.resize(img,(400,400))
    img = img/255
    X_valid.append(img)
X_valid = np.array(X_valid)

# the image is split into 4X4 grid, and each cells contain info about (present, class, center x, center y, box height, box width)
# the dimension of each image is 4X4X6
y_train =[]
for i in tqdm(trainimglist):
    d = traindf[traindf["filename"]==i]
    d = d.sort_values("gridnum")
    d = d.reset_index()
    d = d.drop("index",axis=1)
    label = np.zeros((grid,grid,6))
    for m in range(len(d)):
        bx = d["bx"][m]
        by = d["by"][m]
        bh = d["bh"][m]
        bw = d["bw"][m]
        gx = int(d["gridx"][m])
        gy = int(d["gridy"][m])
        label[gy,gx,0] = 1
        label[gy,gx,1] = 1
        label[gy,gx,2] = bx
        label[gy,gx,3] = by
        label[gy,gx,4] = bh
        label[gy,gx,5] = bw

    y_train.append(label)

y_train = np.array(y_train)

y_valid =[]
for i in tqdm(validimglist):
    d = validdf[validdf["filename"]==i]
    d = d.sort_values("gridnum")
    d = d.reset_index()
    d = d.drop("index",axis=1)
    label = np.zeros((grid,grid,6))
    for m in range(len(d)):
        bx = d["bx"][m]
        by = d["by"][m]
        bh = d["bh"][m]
        bw = d["bw"][m]
        gx = int(d["gridx"][m])
        gy = int(d["gridy"][m])
        label[gy,gx,0] = 1
        label[gy,gx,1] = 1
        label[gy,gx,2] = bx
        label[gy,gx,3] = by
        label[gy,gx,4] = bh
        label[gy,gx,5] = bw

    y_valid.append(label)

y_valid = np.array(y_valid)

# final inspection of the image and coordinates
m=1
img = cv2.imread(filepath + "train/"+trainimglist[m])
img = cv2.resize(img,(400,400))
for i in range(grid):
    cv2.line(img,(int(400/grid)*i,0),(int(400/grid)*i,400),(0,255,0),1)
    cv2.line(img,(0,int(400/grid)*i),(400,int(400/grid)*i),(0,255,0),1)
s = traindf[traindf["filename"]==trainimglist[m]]
s = s.reset_index()
for i in tqdm(range(len(s))):
    xc = np.int(s["xc"][i]*400)
    yc = np.int(s["yc"][i]*400)
    cv2.circle(img,(xc,yc),5,(0,0,255),-1)
    x1 = np.int(s["xmin"][i]*400)
    y1 = np.int(s["ymin"][i]*400)
    x2 = np.int(s["xmax"][i]*400)
    y2 = np.int(s["ymax"][i]*400)
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
plt.imshow(img)


import tensorflow as tf
from tensorflow.keras.regularizers import L2
model = tf.keras.applications.xception.Xception(input_shape=(400,400,3),include_top=False, weights="imagenet")
model.trainable = False
x = model.output
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(filters=256,kernel_size= (3,3),strides= (1,1),kernel_regularizer=L2(l2=0.0005))(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Conv2D(filters=128,kernel_size= (3,3),strides= (1,1),padding="same",kernel_regularizer=L2(l2=0.0005))(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(64,kernel_regularizer=L2(l2=0.0005))(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(6,activation="sigmoid")(x)
model = tf.keras.Model(inputs=model.input, outputs=x)
model.summary()

# custom loss
import keras.backend as K
def custom_loss(y_true, y_pred):
    contrue = y_true[:,:,:,0]
    conpred = y_pred[:,:,:,0]
    classtrue = y_true[:,:,:,1]
    classpred = y_pred[:,:,:,1]
    bbxctrue = y_true[:,:,:,2]
    bbxcpred = y_pred[:,:,:,2]
    bbyctrue = y_true[:,:,:,3]
    bbycpred = y_pred[:,:,:,3]
    bbhtrue = y_true[:,:,:,4]
    bbhpred = y_pred[:,:,:,4]
    bbwtrue = y_true[:,:,:,5]
    bbwpred = y_pred[:,:,:,5]

    # put more weight on 1 than 0, because of imbalance data
    conloss = (((1-contrue)*0.5)*K.square(conpred-contrue))+(contrue*K.square(conpred-contrue))
    classloss = contrue*K.square(classpred-classtrue)
    # put more weight on bbc and bb, because we want the boxes to be in the right shape and size
    bbcloss = contrue*5*(K.square(bbxcpred-bbxctrue)+K.square(bbycpred-bbyctrue))
    bbloss = contrue*5*(K.square(K.sqrt(bbhpred)-K.sqrt(bbhtrue))+K.square(K.sqrt(bbwpred)-K.sqrt(bbwtrue)))
    loss = conloss+classloss+bbcloss+bbloss
    loss = K.sum(loss)
    return loss

opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

import tensorflow.keras.backend as K
def f1_con(y_true, y_pred):
    y_true = y_true[:,:,:,0]
    y_pred = y_pred[:,:,:,0]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
def recall_con(y_true, y_pred):
    y_true = y_true[:,:,:,0]
    y_pred = y_pred[:,:,:,0]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / possible_positives
    return recall
def f1_class(y_true, y_pred):
    y_true = y_true[:,:,:,1]
    y_pred = y_pred[:,:,:,1]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_class = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_class
def recall_class(y_true, y_pred):
    y_true = y_true[:,:,:,1]
    y_pred = y_pred[:,:,:,1]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / possible_positives
    return recall

metrics = [f1_con,recall_con,f1_class,recall_class]

model.compile(loss=custom_loss, optimizer=opt,metrics=metrics)

rlrop = tf.keras.callbacks.ReduceLROnPlateau("loss",0.7,5,verbose=1)

# training model
history = model.fit(X_train, y_train,validation_data=(X_valid, y_valid),batch_size=32,epochs=70,callbacks=rlrop)

"""
Epoch 65/70
5/5 [==============================] - 29s 6s/step - loss: 49.0506 - f1_con: 0.4877 - recall_con: 0.8438 - f1_class: 0.2651 - val_loss: 46.0177 - val_f1_con: 0.4819 - val_recall_con: 0.6897 - val_f1_class: 0.2500 - lr: 5.0000e-04
Epoch 66/70
5/5 [==============================] - 29s 6s/step - loss: 49.1307 - f1_con: 0.5019 - recall_con: 0.8525 - f1_class: 0.2629 - val_loss: 44.7479 - val_f1_con: 0.4494 - val_recall_con: 0.6897 - val_f1_class: 0.2257 - lr: 5.0000e-04
Epoch 67/70
5/5 [==============================] - 30s 6s/step - loss: 46.5921 - f1_con: 0.4901 - recall_con: 0.8199 - f1_class: 0.2682 - val_loss: 47.6432 - val_f1_con: 0.4565 - val_recall_con: 0.7241 - val_f1_class: 0.2222 - lr: 5.0000e-04
Epoch 68/70
5/5 [==============================] - 30s 6s/step - loss: 46.6978 - f1_con: 0.5005 - recall_con: 0.8201 - f1_class: 0.2739 - val_loss: 50.2959 - val_f1_con: 0.4719 - val_recall_con: 0.7241 - val_f1_class: 0.2172 - lr: 5.0000e-04
Epoch 69/70
5/5 [==============================] - 30s 6s/step - loss: 47.8576 - f1_con: 0.5230 - recall_con: 0.8584 - f1_class: 0.2730 - val_loss: 50.8796 - val_f1_con: 0.4390 - val_recall_con: 0.6207 - val_f1_class: 0.2214 - lr: 5.0000e-04
Epoch 70/70
5/5 [==============================] - 29s 6s/step - loss: 45.6633 - f1_con: 0.4811 - recall_con: 0.7921 - f1_class: 0.2630 - val_loss: 53.0683 - val_f1_con: 0.4286 - val_recall_con: 0.6207 - val_f1_class: 0.2283 - lr: 5.0000e-04
"""

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()

plt.subplot(4,2,1)
plt.plot(history.history['f1_con'], label='f1_con')
plt.plot(history.history['val_f1_con'], label = 'val_f1_con')
plt.xlabel('Epoch')
plt.ylabel('f1_con')
plt.legend()
plt.subplot(4,2,2)
plt.plot(history.history['recall_con'], label='recall_con')
plt.plot(history.history['val_recall_con'], label = 'val_recall_con')
plt.xlabel('Epoch')
plt.ylabel('recall_con')
plt.legend()
plt.subplot(4,2,3)
plt.plot(history.history['f1_class'], label='f1_class')
plt.plot(history.history['val_f1_class'], label = 'val_f1_class')
plt.xlabel('Epoch')
plt.ylabel('f1_class')
plt.legend()
plt.subplot(4,2,4)
plt.plot(history.history['recall_class'], label='recall_class')
plt.plot(history.history['val_recall_class'], label = 'val_recall_class')
plt.xlabel('Epoch')
plt.ylabel('recall_class')
plt.legend()




# preprocess test data as did with train data and valid data
testdf = pd.read_csv(filepath + "test/_annotations.csv")

testdf["xc"] = 0
testdf["yc"] = 0
for i in tqdm(range(len(testdf))):
    x = (testdf["xmax"][i]+testdf["xmin"][i])/2
    y = (testdf["ymax"][i]+testdf["ymin"][i])/2
    testdf["xc"][i] = x
    testdf["yc"][i] = y


w = testdf["width"]
h = testdf["height"]
testdf["xmin"] = testdf["xmin"]/w
testdf["ymin"] = testdf["ymin"]/h
testdf["xmax"] = testdf["xmax"]/w
testdf["ymax"] = testdf["ymax"]/h
testdf["xc"] = testdf["xc"]/w
testdf["yc"] = testdf["yc"]/h


grid = 4

m=0
img = cv2.imread(filepath + "test/"+testdf["filename"][m])
img = cv2.resize(img,(400,400))
for i in range(grid):
    cv2.line(img,(int(400/grid)*i,0),(int(400/grid)*i,400),(0,255,0),1)
    cv2.line(img,(0,int(400/grid)*i),(400,int(400/grid)*i),(0,255,0),1)
s = testdf[testdf["filename"]==testdf["filename"][m]]
s = s.reset_index()
for i in tqdm(range(len(s))):
    xc = np.int(s["xc"][i]*400)
    yc = np.int(s["yc"][i]*400)
    cv2.circle(img,(xc,yc),5,(0,0,255),-1)
    x1 = np.int(s["xmin"][i]*400)
    y1 = np.int(s["ymin"][i]*400)
    x2 = np.int(s["xmax"][i]*400)
    y2 = np.int(s["ymax"][i]*400)
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
plt.imshow(img)


grid = 4

testdf["gridx"] = np.floor(testdf["xc"]*grid)
testdf["gridy"] = np.floor(testdf["yc"]*grid)

testdf["gridnum"] = testdf["gridx"]+testdf["gridy"]*grid

# bx,by
testdf["bx"] = (testdf["xc"]-(testdf["gridx"]*int(400/grid)/400))/(int(400/grid)/400)
testdf["by"] = (testdf["yc"]-(testdf["gridy"]*int(400/grid)/400))/(int(400/grid)/400)

# bw, bh
testdf["bw"] = testdf["xmax"]-testdf["xmin"]
testdf["bh"] = testdf["ymax"]-testdf["ymin"]


testimglist = testdf["filename"].unique().tolist()

X_test = []
t=0
for i in tqdm(testimglist):
    img = cv2.imread(filepath+"test/"+i)
    img = cv2.resize(img,(400,400))
    img = img/255
    X_test.append(img)
X_test = np.array(X_test)


y_test =[]
for i in tqdm(testimglist):
    d = testdf[testdf["filename"]==i]
    d = d.sort_values("gridnum")
    d = d.reset_index()
    d = d.drop("index",axis=1)
    label = np.zeros((grid,grid,6))
    for m in range(len(d)):
        bx = d["bx"][m]
        by = d["by"][m]
        bh = d["bh"][m]
        bw = d["bw"][m]
        gx = int(d["gridx"][m])
        gy = int(d["gridy"][m])
        label[gy,gx,0] = 1
        label[gy,gx,1] = 1
        label[gy,gx,2] = bx
        label[gy,gx,3] = by
        label[gy,gx,4] = bh
        label[gy,gx,5] = bw

    y_test.append(label)

y_test = np.array(y_test)

# final inspection of the test image and coordinates
m=1
img = cv2.imread(filepath + "test/"+testimglist[m])
img = cv2.resize(img,(400,400))
for i in range(grid):
    cv2.line(img,(int(400/grid)*i,0),(int(400/grid)*i,400),(0,255,0),1)
    cv2.line(img,(0,int(400/grid)*i),(400,int(400/grid)*i),(0,255,0),1)
s = testdf[testdf["filename"]==testimglist[m]]
s = s.reset_index()
for i in tqdm(range(len(s))):
    xc = np.int(s["xc"][i]*400)
    yc = np.int(s["yc"][i]*400)
    cv2.circle(img,(xc,yc),5,(0,0,255),-1)
    x1 = np.int(s["xmin"][i]*400)
    y1 = np.int(s["ymin"][i]*400)
    x2 = np.int(s["xmax"][i]*400)
    y2 = np.int(s["ymax"][i]*400)
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
plt.imshow(img)

# prediction
y_pred = model.predict(X_test)

# retrieve any grid with the [probability(present)*probability(class)] greater than 0.5. If none, retrieve the highest probability instead.
y_pred_list = []
for i in tqdm(range(len(y_pred))):
    y_pred_i = []
    for m in range(4):
        for n in range(4):
            if y_pred[i,m,n,0] >= 0.5:
                con = y_pred[i,m,n,0]*y_pred[i,m,n,1]
                bx = (n+y_pred[i,m,n,2])*(400/grid)
                by = (m+y_pred[i,m,n,3])*(400/grid)
                bh = y_pred[i,m,n,4]*400
                bw = y_pred[i,m,n,5]*400
                xmin = np.maximum(0,bx-bw/2)
                xmax = np.maximum(0,bx+bw/2)
                ymin = np.maximum(0,by-bh/2)
                ymax = np.maximum(0,by+bh/2)
                y_pred_i.append([con,np.round(bx),np.round(by),np.round(xmin),np.round(ymin),np.round(xmax),np.round(ymax)])
    if y_pred_i == []:
        n = y_pred[i][:,:,0].argmax()
        r = int(np.floor(n/4))
        c = int(n-r*4)
        con = y_pred[i,r,c,0]*y_pred[i,r,c,1]
        bx = (c+y_pred[i,r,c,2])*(400/grid)
        by = (r+y_pred[i,r,c,3])*(400/grid)
        bh = y_pred[i,r,c,4]*400
        bw = y_pred[i,r,c,5]*400
        xmin = np.maximum(0,bx-bw/2)
        xmax = np.maximum(0,bx+bw/2)
        ymin = np.maximum(0,by-bh/2)
        ymax = np.maximum(0,by+bh/2)
        y_pred_i.append([con,np.round(bx),np.round(by),np.round(xmin),np.round(ymin),np.round(xmax),np.round(ymax)])

    y_pred_list.append(y_pred_i)

# compare true test coordinates with predicted coordinates (the highest confidence)
m=5
img = cv2.imread(filepath + "test/"+testimglist[m])
img = cv2.resize(img,(400,400))
for i in range(grid):
    cv2.line(img,(int(400/grid)*i,0),(int(400/grid)*i,400),(0,255,0),1)
    cv2.line(img,(0,int(400/grid)*i),(400,int(400/grid)*i),(0,255,0),1)
s = testdf[testdf["filename"]==testimglist[m]]
s = s.reset_index()
for i in tqdm(range(len(s))):
    xc = np.int(s["xc"][i]*400)
    yc = np.int(s["yc"][i]*400)
    cv2.circle(img,(xc,yc),5,(0,0,255),-1)
    x1 = np.int(s["xmin"][i]*400)
    y1 = np.int(s["ymin"][i]*400)
    x2 = np.int(s["xmax"][i]*400)
    y2 = np.int(s["ymax"][i]*400)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
l=[]
for i in y_pred_list[m]:
    l.append(i[0])
n = np.argmax(l)
bx = int(y_pred_list[m][n][1])
by = int(y_pred_list[m][n][2])
cv2.circle(img,(bx,by),5,(255,0,0),-1)
x1 = int(y_pred_list[m][n][3])
y1 = int(y_pred_list[m][n][4])
x2 = int(y_pred_list[m][n][5])
y2 = int(y_pred_list[m][n][6])
cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
conpercent = np.round(y_pred_list[m][n][0]*100,3)
cv2.putText(img, "{}%".format(conpercent), (x1+5,y1+15), fontFace=1,fontScale=1, color=(255,0,0), thickness=2)

plt.imshow(img)


# showing image with just predicted data
m=5
img = cv2.imread(filepath + "test/"+testimglist[m])
img = cv2.resize(img,(400,400))
l=[]
for i in y_pred_list[m]:
    l.append(i[0])
n = np.argmax(l)
bx = int(y_pred_list[m][n][1])
by = int(y_pred_list[m][n][2])
cv2.circle(img,(bx,by),5,(255,0,0),-1)
x1 = int(y_pred_list[m][n][3])
y1 = int(y_pred_list[m][n][4])
x2 = int(y_pred_list[m][n][5])
y2 = int(y_pred_list[m][n][6])
cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
conpercent = np.round(y_pred_list[m][n][0]*100,3)
cv2.putText(img, "{}%".format(conpercent), (x1+5,y1+15), fontFace=1,fontScale=1, color=(255,0,0), thickness=2)
plt.imshow(img)



# plotting all test images with predicted coordinates
import cv2
for m in range(len(y_pred_list)):
    plt.subplot(5,4,m+1)
    img = cv2.imread(filepath + "test/"+testimglist[m])
    img = cv2.resize(img,(400,400))
    l=[]
    for i in y_pred_list[m]:
        l.append(i[0])
    n = np.argmax(l)
    bx = int(y_pred_list[m][n][1])
    by = int(y_pred_list[m][n][2])
    cv2.circle(img,(bx,by),5,(255,0,0),-1)
    x1 = int(y_pred_list[m][n][3])
    y1 = int(y_pred_list[m][n][4])
    x2 = int(y_pred_list[m][n][5])
    y2 = int(y_pred_list[m][n][6])
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
    conpercent = np.round(y_pred_list[m][n][0]*100,3)
    cv2.putText(img, "{}%".format(conpercent), (x1+5,y1+15), fontFace=1,fontScale=1, color=(255,0,0), thickness=2)
    plt.imshow(img)
