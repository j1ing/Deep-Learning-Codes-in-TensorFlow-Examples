import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# load data
filepath = ".../American Sign Language Letters.v1-v1.tensorflow/"

traindf = pd.read_csv(filepath + "train/_annotations.csv")
validdf = pd.read_csv(filepath + "valid/_annotations.csv")

# normalize bounding box coordinates into between 0 and 1
image_width = 416
image_height = 416

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


# counts for each letter
traindf["class"].value_counts()
"""
J    78
I    78
L    72
A    69
S    69
Z    66
E    63
X    63
D    63
F    60
Q    60
G    60
N    60
W    57
V    57
O    54
C    54
P    51
H    51
M    51
K    51
Y    48
U    48
R    48
T    42
B    39
"""

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


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(traindf["class"])

traindf["classnum"] = 0
for i in range(len(traindf)):
    traindf["classnum"][i] = integer_encoded[i]

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(validdf["class"])

validdf["classnum"] = 0
for i in range(len(validdf)):
    validdf["classnum"][i] = integer_encoded[i]

# In this particular dataset, I split the bounding boxes training and classification into two seperate training model due to the difficulty in image classification in this dataset.
# each class has around 50 - 60 images. Without more images and augmentation, I have a tough time using Xception Net and moblie net to achieve good accuracy.
# I tried to implement the class labels with the bounding boxes to make the dimension of each image 4X4X31. (present, bx, by, bh, bw, c1....c26).
# the training model has a very difficult time classifying. To avoid overfitting in the bounding box portion, I therefore separated the training model into two.
# the image is split into 4X4 grid, and each cells contain info about (present, class, center x, center y, box height, box width)
# the dimension of each image is 4X4X5
y_train_bb =[]
for i in tqdm(trainimglist):
    d = traindf[traindf["filename"]==i]
    d = d.sort_values("gridnum")
    d = d.reset_index()
    d = d.drop("index",axis=1)
    label = np.zeros((grid,grid,5))
    for m in range(len(d)):
        bx = d["bx"][m]
        by = d["by"][m]
        bh = d["bh"][m]
        bw = d["bw"][m]
        gx = int(d["gridx"][m])
        gy = int(d["gridy"][m])
        label[gy,gx,0] = 1
        label[gy,gx,1] = bx
        label[gy,gx,2] = by
        label[gy,gx,3] = bh
        label[gy,gx,4] = bw

    y_train_bb.append(label)

y_train_bb = np.array(y_train_bb)

y_train_label = []
for i in traindf["classnum"]:
    y_train_label.append(i)
y_train_label = np.array(y_train_label)

y_valid_bb =[]
for i in tqdm(validimglist):
    d = validdf[validdf["filename"]==i]
    d = d.sort_values("gridnum")
    d = d.reset_index()
    d = d.drop("index",axis=1)
    label = np.zeros((grid,grid,5))
    for m in range(len(d)):
        bx = d["bx"][m]
        by = d["by"][m]
        bh = d["bh"][m]
        bw = d["bw"][m]
        gx = int(d["gridx"][m])
        gy = int(d["gridy"][m])
        label[gy,gx,0] = 1
        label[gy,gx,1] = bx
        label[gy,gx,2] = by
        label[gy,gx,3] = bh
        label[gy,gx,4] = bw

    y_valid_bb.append(label)

y_valid_bb = np.array(y_valid_bb)

y_valid_label = []
for i in validdf["classnum"]:
    y_valid_label.append(i)
y_valid_label = np.array(y_valid_label)


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
    cv2.putText(img, "{}{}".format(traindf["class"][m],traindf["classnum"][m]), (x1+5,y1+15), fontFace=1,fontScale=1, color=(255,0,0), thickness=2)
plt.imshow(img)



m=1
img = cv2.imread(filepath + "valid/"+validimglist[m])
img = cv2.resize(img,(400,400))
for i in range(grid):
    cv2.line(img,(int(400/grid)*i,0),(int(400/grid)*i,400),(0,255,0),1)
    cv2.line(img,(0,int(400/grid)*i),(400,int(400/grid)*i),(0,255,0),1)
s = validdf[validdf["filename"]==validimglist[m]]
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
    cv2.putText(img, "{}{}".format(validdf["class"][m],validdf["classnum"][m]), (x1+5,y1+15), fontFace=1,fontScale=1, color=(255,0,0), thickness=2)
plt.imshow(img)


# build model bounding box
import tensorflow as tf
from tensorflow.keras.regularizers import L2
modelbb = tf.keras.applications.xception.Xception(input_shape=(400,400,3),include_top=False, weights="imagenet")
modelbb.trainable = False
x = modelbb.output

x1 = tf.keras.layers.MaxPooling2D((2,2))(x)
x1 = tf.keras.layers.Conv2D(filters=512,kernel_size= (3,3),strides= (1,1),kernel_regularizer=L2(l2=0.0005))(x1)
x1 = tf.keras.layers.LeakyReLU()(x1)
x1 = tf.keras.layers.BatchNormalization()(x1)
x1 = tf.keras.layers.Dropout(0.5)(x1)
x1 = tf.keras.layers.Conv2D(filters=256,kernel_size= (3,3),strides= (1,1),padding="same",kernel_regularizer=L2(l2=0.0005))(x1)
x1 = tf.keras.layers.LeakyReLU()(x1)
x1 = tf.keras.layers.BatchNormalization()(x1)
x1 = tf.keras.layers.Dropout(0.5)(x1)
x1 = tf.keras.layers.Dense(128,kernel_regularizer=L2(l2=0.0005))(x1)
x1 = tf.keras.layers.LeakyReLU()(x1)
x1 = tf.keras.layers.BatchNormalization()(x1)
x1 = tf.keras.layers.Dropout(0.5)(x1)
x1 = tf.keras.layers.Dense(5,name="bb",activation="sigmoid")(x1)

modelbb = tf.keras.Model(inputs=modelbb.input, outputs=x1)
modelbb.summary()

# custom loss
import keras.backend as K
def custom_loss(y_true, y_pred):
    contrue = y_true[:,:,:,0]
    conpred = y_pred[:,:,:,0]
    bbxctrue = y_true[:,:,:,1]
    bbxcpred = y_pred[:,:,:,1]
    bbyctrue = y_true[:,:,:,2]
    bbycpred = y_pred[:,:,:,2]
    bbhtrue = y_true[:,:,:,3]
    bbhpred = y_pred[:,:,:,3]
    bbwtrue = y_true[:,:,:,4]
    bbwpred = y_pred[:,:,:,4]
    # put more weight on 1 than 0, because of imbalance data
    conloss = (((1-contrue)*0.5)*K.square(conpred-contrue))+(contrue*K.square(conpred-contrue))
    # put more weight on bbc and bb, because we want the boxes to be in the right shape and size
    bbcloss = contrue*5*(K.square(bbxcpred-bbxctrue)+K.square(bbycpred-bbyctrue))
    bbloss = contrue*5*(K.square(K.sqrt(bbhpred)-K.sqrt(bbhtrue))+K.square(K.sqrt(bbwpred)-K.sqrt(bbwtrue)))
    loss = conloss+bbcloss+bbloss
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
metrics = [f1_con,recall_con]

modelbb.compile(loss=custom_loss, optimizer=opt,metrics=metrics)

rlrop = tf.keras.callbacks.ReduceLROnPlateau("loss",0.7,5,verbose=1)


# training model bounding boxes
history = modelbb.fit(X_train, y_train_bb,validation_data=(X_valid, y_valid_bb),batch_size=32,epochs=10,callbacks=rlrop)

"""
Epoch 5/10
48/48 [==============================] - 307s 6s/step - loss: 55.9846 - f1_con: 0.4643 - recall_con: 0.8073 - val_loss: 46.9167 - val_f1_con: 0.5590 - val_recall_con: 0.8500 - lr: 5.0000e-04
Epoch 6/10
48/48 [==============================] - 291s 6s/step - loss: 48.4874 - f1_con: 0.4977 - recall_con: 0.8092 - val_loss: 37.7703 - val_f1_con: 0.5924 - val_recall_con: 0.8625 - lr: 5.0000e-04
Epoch 7/10
48/48 [==============================] - 331s 7s/step - loss: 42.7576 - f1_con: 0.5203 - recall_con: 0.8021 - val_loss: 33.0535 - val_f1_con: 0.5636 - val_recall_con: 0.7563 - lr: 5.0000e-04
Epoch 8/10
48/48 [==============================] - 297s 6s/step - loss: 37.2085 - f1_con: 0.5675 - recall_con: 0.8151 - val_loss: 30.2048 - val_f1_con: 0.5821 - val_recall_con: 0.7875 - lr: 5.0000e-04
Epoch 9/10
48/48 [==============================] - 297s 6s/step - loss: 33.6462 - f1_con: 0.5867 - recall_con: 0.8197 - val_loss: 28.8359 - val_f1_con: 0.5691 - val_recall_con: 0.7000 - lr: 5.0000e-04
Epoch 10/10
48/48 [==============================] - 320s 7s/step - loss: 30.5866 - f1_con: 0.6126 - recall_con: 0.8060 - val_loss: 27.0671 - val_f1_con: 0.6065 - val_recall_con: 0.8000 - lr: 5.0000e-04
"""

# building model classifying letters
import tensorflow as tf
from tensorflow.keras.regularizers import L2
modellabel = tf.keras.applications.mobilenet.MobileNet(input_shape=(400,400,3),include_top=False, weights="imagenet")
modellabel.trainable = False
x = modellabel.output

x2 = tf.keras.layers.Flatten()(x)

x2 = tf.keras.layers.Dense(26,name="letter",activation="softmax")(x2)
modellabel = tf.keras.Modellabel(inputs=modellabel.input, outputs=x2)
modellabel.summary()

losses = {"letter": "sparse_categorical_crossentropy"}

opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

metrics = ["accuracy"]

model.compile(loss=losses, optimizer=opt,metrics=metrics)

rlrop = tf.keras.callbacks.ReduceLROnPlateau("loss",0.7,5,verbose=1)


# training model
history = modellabel.fit(X_train, y_train_label,validation_data=(X_valid, y_valid_label),batch_size=32,epochs=5,callbacks=rlrop)

"""
Epoch 1/5
48/48 [==============================] - 52s 962ms/step - loss: 8.9557 - accuracy: 0.5324 - val_loss: 6.5489 - val_accuracy: 0.5764 - lr: 5.0000e-04
Epoch 2/5
48/48 [==============================] - 47s 971ms/step - loss: 0.3888 - accuracy: 0.9630 - val_loss: 6.0127 - val_accuracy: 0.6389 - lr: 5.0000e-04
Epoch 3/5
48/48 [==============================] - 48s 992ms/step - loss: 0.1433 - accuracy: 0.9788 - val_loss: 8.7436 - val_accuracy: 0.5486 - lr: 5.0000e-04
Epoch 4/5
48/48 [==============================] - 48s 997ms/step - loss: 0.0646 - accuracy: 0.9861 - val_loss: 7.9534 - val_accuracy: 0.6181 - lr: 5.0000e-04
Epoch 5/5
48/48 [==============================] - 48s 1s/step - loss: 0.0293 - accuracy: 0.9940 - val_loss: 7.4697 - val_accuracy: 0.5903 - lr: 5.0000e-04
"""


# load test data
testdf = pd.read_csv(filepath + "test/_annotations.csv")


testimglist = testdf["filename"].unique().tolist()

X_test = []
t=0
for i in tqdm(testimglist):
    img = cv2.imread(filepath+"test/"+i)
    img = cv2.resize(img,(400,400))
    img = img/255
    X_test.append(img)
X_test = np.array(X_test)

# predict test images
y_pred_label = modellabel.predict(X_test)

labelpred = []
labelprob = []

for i in y_pred_label:
    idx = i.argmax()
    labelpred.append(idx)
    prob = np.round(i[idx]*100,2)
    labelprob.append(prob)

labelpred = label_encoder.inverse_transform(labelpred)

labeltrue = []
for i in testdf["class"]:
    labeltrue.append(i)


y_pred_bb = modelbb.predict(X_test)

grid =4

bb_pred = []
for i in tqdm(range(len(y_pred_bb))):
    y_pred_bb_i = []
    for m in range(4):
        for n in range(4):
            if y_pred_bb[i,m,n,0] >= 0.5:
                con = y_pred_bb[i,m,n,0]
                bx = (n+y_pred_bb[i,m,n,1])*(400/grid)
                by = (m+y_pred_bb[i,m,n,2])*(400/grid)
                bh = y_pred_bb[i,m,n,3]*400
                bw = y_pred_bb[i,m,n,4]*400
                xmin = np.maximum(0,bx-bw/2)
                xmax = np.maximum(0,bx+bw/2)
                ymin = np.maximum(0,by-bh/2)
                ymax = np.maximum(0,by+bh/2)
                y_pred_bb_i.append([con,np.round(bx),np.round(by),np.round(xmin),np.round(ymin),np.round(xmax),np.round(ymax)])
    if y_pred_bb_i == []:
        n = y_pred_bb[i][:,:,0].argmax()
        r = int(np.floor(n/grid))
        c = int(n-r*grid)
        con = y_pred_bb[i,r,c,0]
        bx = (c+y_pred_bb[i,r,c,1])*(400/grid)
        by = (r+y_pred_bb[i,r,c,2])*(400/grid)
        bh = y_pred_bb[i,r,c,3]*400
        bw = y_pred_bb[i,r,c,4]*400
        xmin = np.maximum(0,bx-bw/2)
        xmax = np.maximum(0,bx+bw/2)
        ymin = np.maximum(0,by-bh/2)
        ymax = np.maximum(0,by+bh/2)
        y_pred_bb_i.append([con,np.round(bx),np.round(by),np.round(xmin),np.round(ymin),np.round(xmax),np.round(ymax)])

    bb_pred.append(y_pred_bb_i)


# plotting the test images with predicted bounding boxes and classfication    
    
m=5
img = cv2.imread(filepath + "test/"+testimglist[m])
img = cv2.resize(img,(400,400))
l=[]
for i in bb_pred[m]:
    l.append(i[0])
n = np.argmax(l)
bx = int(bb_pred[m][n][1])
by = int(bb_pred[m][n][2])
cv2.circle(img,(bx,by),5,(255,0,0),-1)
x1 = int(bb_pred[m][n][3])
y1 = int(bb_pred[m][n][4])
x2 = int(bb_pred[m][n][5])
y2 = int(bb_pred[m][n][6])
cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
cv2.putText(img, "True: {} Predict:{} {}%".format(labeltrue[m],labelpred[m],labelprob[m]), (x1+5,y1+15), fontFace=1,fontScale=1, color=(255,0,0), thickness=2)
plt.imshow(img)


# plot random test images
sample = np.random.randint(0,72,9)
"""
array([30, 19, 43, 44, 19, 64,  4,  1,  3])
"""

for m in range(len(sample)):
    plt.subplot(3,3,m+1)
    m = sample[m]
    img = cv2.imread(filepath + "test/"+testimglist[m])
    img = cv2.resize(img,(400,400))
    l=[]
    for i in bb_pred[m]:
        l.append(i[0])
    n = np.argmax(l)
    bx = int(bb_pred[m][n][1])
    by = int(bb_pred[m][n][2])
    cv2.circle(img,(bx,by),5,(255,0,0),-1)
    x1 = int(bb_pred[m][n][3])
    y1 = int(bb_pred[m][n][4])
    x2 = int(bb_pred[m][n][5])
    y2 = int(bb_pred[m][n][6])
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.putText(img, "True: {} Predict:{} {}%".format(labeltrue[m],labelpred[m],labelprob[m]), (x1+5,y1+15), fontFace=1,fontScale=1, color=(255,0,0), thickness=2)
    plt.imshow(img)

    
# the increase number of training images definitely boosted the accuracy of the bounding boxes. This training dataset has a total of 1512 images. 
# it has a lot more than the racoon dataset (150 images)
