import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
filepath = "/Users/jeffling/Downloads/American Sign Language Letters.v1-v1.tensorflow/"

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

# finalize the dataframe
trainfinal = pd.concat([traindf["filename"],traindf["class"],xtraindf,ytraindf],axis=1)
validfinal = pd.concat([validdf["filename"],validdf["class"],xvaliddf,yvaliddf],axis=1)

# counts for each letter
trainfinal["class"].value_counts()
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

# split dataframe into X(images) and y(letters and bounding box coordinates)
trainimg= pd.DataFrame(trainfinal["filename"],columns=["filename"])
trainclass = trainfinal["class"]
trainbb = trainfinal[["xmin","ymin","xmax","ymax"]]

validimg= pd.DataFrame(validfinal["filename"],columns=["filename"])
validclass = validfinal["class"]
validbb = validfinal[["xmin","ymin","xmax","ymax"]]

# format X and y into numpy array
# training images
import cv2

trainimgarr = []

from tqdm import tqdm
for i in tqdm(range(len(trainimg))):
    im = cv2.imread(filepath+"train/"+trainimg["filename"][i])
    im = cv2.resize(im, (256,256))
    im = im/255
    trainimgarr.append(im)

trainimgarr = np.array(trainimgarr)

# training class
trainclassarr = []

for i in tqdm(trainclass):
    trainclassarr.append(i)

trainclassarr = np.array(trainclassarr)

# turn letters into one hot encoders
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(trainclassarr)
print(integer_encoded)

from tensorflow.keras.utils import to_categorical
trainclassarr = to_categorical(integer_encoded)

# training bounding box
trainbbarr = []

for i in tqdm(range(len(trainbb))):
    xmin = trainbb["xmin"][i]
    ymin = trainbb["ymin"][i]
    xmax = trainbb["xmax"][i]
    ymax = trainbb["ymax"][i]
    labellist = [xmin,ymin,xmax,ymax]
    trainbbarr.append(labellist)

trainbbarr = np.array(trainbbarr)

# validation images
validimgarr = []

from tqdm import tqdm
for i in tqdm(range(len(validimg))):
    im = cv2.imread(filepath+"valid/"+validimg["filename"][i])
    im = cv2.resize(im, (256,256))
    im = im/255
    validimgarr.append(im)

validimgarr = np.array(validimgarr)

# validation letters
validclassarr = []

for i in tqdm(validclass):
    validclassarr.append(i)

validclassarr = np.array(validclassarr)

# turn letters into one hot encoders
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(validclassarr)
print(integer_encoded)

from tensorflow.keras.utils import to_categorical
validclassarr = to_categorical(integer_encoded)

# validation bounding box
validbbarr = []

for i in tqdm(range(len(validbb))):
    xmin = validbb["xmin"][i]
    ymin = validbb["ymin"][i]
    xmax = validbb["xmax"][i]
    ymax = validbb["ymax"][i]
    labellist = [xmin,ymin,xmax,ymax]
    validbbarr.append(labellist)

validbbarr = np.array(validbbarr)

# build model
import tensorflow as tf

model = tf.keras.applications.mobilenet.MobileNet(input_shape=(256,256,3),include_top=False,weights="imagenet")

model.trainable = False

x = tf.keras.layers.Flatten()(model.output)

# letter classification
c = tf.keras.layers.Dense(1024, activation='relu',activity_regularizer=tf.keras.regularizers.L2(1e-5))(x)
c = tf.keras.layers.BatchNormalization()(c)
c = tf.keras.layers.Dropout(0.7)(c)
c = tf.keras.layers.Dense(512, activation='relu',activity_regularizer=tf.keras.regularizers.L2(1e-5))(c)
c = tf.keras.layers.BatchNormalization()(c)
c = tf.keras.layers.Dropout(0.6)(c)
c = tf.keras.layers.Dense(26,name="class", activation='softmax')(c)

# bounding box
bb = tf.keras.layers.Dense(128, activation='relu')(x)
bb = tf.keras.layers.Dropout(0.2)(bb)
bb = tf.keras.layers.Dense(64, activation='relu')(bb)
bb = tf.keras.layers.Dropout(0.1)(bb)
bb = tf.keras.layers.Dense(32, activation='relu')(bb)
bb = tf.keras.layers.Dropout(0.2)(bb)
bb = tf.keras.layers.Dense(16, activation='relu')(bb)
bb = tf.keras.layers.Dropout(0.1)(bb)
bb = tf.keras.layers.Dense(4,name="bounding_box", activation='sigmoid')(bb)


model = tf.keras.Model(inputs=model.input, outputs=(c,bb))

# compilt model
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
losses = {"class": "categorical_crossentropy","bounding_box": "mean_squared_error"}
lossWeights = {"class": 1.0,"bounding_box": 0.5}

model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)

rlrop = tf.keras.callbacks.ReduceLROnPlateau("class_loss",0.7,5,verbose=1)

# combine letters and bounding box together as y
traintargets = {"class": trainclassarr,"bounding_box": trainbbarr}
validtargets = {"class": validclassarr,"bounding_box": validbbarr}

# training model
history = model.fit(trainimgarr, traintargets,validation_data=(validimgarr, validtargets),batch_size=16,epochs=30,callbacks=rlrop)


# plot total loss, class loss and bounding box loss
plt.subplot(3,1,1)
plt.plot(history.history['class_accuracy'], label='class_accuracy')
plt.plot(history.history['val_class_accuracy'], label = 'val_class_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Class accuracy')
plt.legend()

plt.subplot(3,1,2)
plt.plot(history.history['class_loss'], label='class_loss')
plt.plot(history.history['val_class_loss'], label = 'val_class_loss')
plt.xlabel('Epoch')
plt.ylabel('Class loss')
plt.legend()

plt.subplot(3,1,3)
plt.plot(history.history['bounding_box_loss'], label='bounding_box_loss')
plt.plot(history.history['val_bounding_box_loss'], label = 'val_bounding_box_loss')
plt.xlabel('Epoch')
plt.ylabel('Bounding bBox loss')
plt.legend()

"""
Epoch 25/30
95/95 [==============================] - 143s 2s/step - loss: 0.3588 - class_loss: 0.1806 - bounding_box_loss: 0.0136 - class_accuracy: 0.9494 - bounding_box_accuracy: 0.6160 - val_loss: 1.1062 - val_class_loss: 1.0040 - val_bounding_box_loss: 0.0088 - val_class_accuracy: 0.7292 - val_bounding_box_accuracy: 0.7014
Epoch 26/30
95/95 [==============================] - 143s 2s/step - loss: 0.3833 - class_loss: 0.2000 - bounding_box_loss: 0.0138 - class_accuracy: 0.9476 - bounding_box_accuracy: 0.6191 - val_loss: 1.1654 - val_class_loss: 1.0678 - val_bounding_box_loss: 0.0098 - val_class_accuracy: 0.7292 - val_bounding_box_accuracy: 0.7222
Epoch 27/30
95/95 [==============================] - 154s 2s/step - loss: 0.3763 - class_loss: 0.2004 - bounding_box_loss: 0.0130 - class_accuracy: 0.9398 - bounding_box_accuracy: 0.6326 - val_loss: 1.1414 - val_class_loss: 1.0448 - val_bounding_box_loss: 0.0092 - val_class_accuracy: 0.7083 - val_bounding_box_accuracy: 0.7153
Epoch 28/30
95/95 [==============================] - 149s 2s/step - loss: 0.3931 - class_loss: 0.2114 - bounding_box_loss: 0.0128 - class_accuracy: 0.9458 - bounding_box_accuracy: 0.6344 - val_loss: 1.1990 - val_class_loss: 1.0971 - val_bounding_box_loss: 0.0093 - val_class_accuracy: 0.7153 - val_bounding_box_accuracy: 0.6944
Epoch 29/30
95/95 [==============================] - 152s 2s/step - loss: 0.4013 - class_loss: 0.2167 - bounding_box_loss: 0.0135 - class_accuracy: 0.9359 - bounding_box_accuracy: 0.6255 - val_loss: 1.1413 - val_class_loss: 1.0476 - val_bounding_box_loss: 0.0099 - val_class_accuracy: 0.7083 - val_bounding_box_accuracy: 0.7153
Epoch 30/30
95/95 [==============================] - 150s 2s/step - loss: 0.4097 - class_loss: 0.2243 - bounding_box_loss: 0.0125 - class_accuracy: 0.9400 - bounding_box_accuracy: 0.6339 - val_loss: 1.1510 - val_class_loss: 1.0413 - val_bounding_box_loss: 0.0107 - val_class_accuracy: 0.7153 - val_bounding_box_accuracy: 0.7431
"""

# load test data
testdf = pd.read_csv(filepath + "test/_annotations.csv")

# turn test images into numpy array
testimgarr = []

from tqdm import tqdm
for i in tqdm(range(len(testdf))):
    im = cv2.imread(filepath+"test/"+testdf["filename"][i])
    im = cv2.resize(im, (256,256))
    im = im/255
    testimgarr.append(im)

testimgarr = np.array(testimgarr)

# predict test images
y_pred = model.predict(testimgarr)

# retrieve predicted classification
pred_test_label = y_pred[0]
# predicted label
testlabel = []
for i in tqdm(range(len(pred_test_label))):
    testlabel.append(pred_test_label[i].argmax())
# predicted label's probability
testlabelprob = []
for i in tqdm(range(len(pred_test_label))):
    testlabelprob.append(np.round((pred_test_label[i].max()*100),2))

# reverse integer class indices into letters
testletters = label_encoder.inverse_transform(testlabel)

# retrieve predicted bounding box coordinates
pred_test_bb = y_pred[1]
# rescale bounding to desired image size
test_bb = np.round(pred_test_bb*416).astype("int64")

# plot images with true bounding box and letter in red, and predicted bounding box and letter in green
import cv2
for i in range(1,13):
    plt.subplot(4,3,i)
    s = np.random.randint(0,len(testdf))
    img = cv2.imread(filepath + "test/" + testdf["filename"][s])
    a = cv2.rectangle(img,(testdf["xmin"][s],testdf["ymin"][s]),(testdf["xmax"][s],testdf["ymax"][s]),(255,0,0),2)
    p = cv2.rectangle(a,(test_bb[s][0],test_bb[s][1]),(test_bb[s][2],test_bb[s][3]),(0,255,0),2)
    indicator = "{} {}%".format(testletters[s],testlabelprob[s])
    z = cv2.putText(p, indicator, (test_bb[s][0], test_bb[s][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    f = cv2.putText(z, testdf["class"][s], (testdf["xmin"][s],testdf["ymin"][s]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,0, 0), 2)
    plt.imshow(f)






