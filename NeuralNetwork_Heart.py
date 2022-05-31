import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
 Heart Attack Analysis & Prediction Dataset
 https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset?select=o2Saturation.csv
"""

"""
About this dataset

age           - Age in Years

sex           - Sex of the Patient

1 = Male
2 = Female
cp             - Chest Pain Type
1 = Typical angina
2 = Atypical angina
3 = Non-anginal pain
trtbps - Resting Blood Pressure (in mm Hg on admission to the hospital)

chol         - Serum Cholesterol in mg/dl

fbs           - Fasting Blood sugar > 120 mg/dl

0 = False
1 = True
restecg   - Resting Electrocardiographic Results
0 = Hypertrophy
1 = Normal
2 = Having ST-T wave abnormality
thalachh   - Maximum Heart Rate Achieved

exng       - Exercise Induced Angina

0 = No
1 = Yes
oldpeak   - ST Depression Induced by Exercise Relative to Rest

slp       - The Slope of the Peak Exercise ST Segment

0 = Downsloping
1 = Flat
2 = Upsloping
caa             - Number of Major Vessels (0-3) Colored by Flourosopy

thall         - Thallium Stress Test Result

0 = Null
1 = Fixed defect
2 = Normal
3 = Reversible defect
output     - The Predicted Attribute - Diagnosis of Heart Disease (angiographic disease status)
0 = < 50% Diameter Narrowing (Heart Attack= No )
1 = > 50% Diameter Narrowing (Heart Attack= Yes)
"""

df = pd.read_csv("/Users/jeffling/Downloads/archive/heart.csv")

df.columns
"""
Index(['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output'],
      dtype='object')
"""

df.info()
"""
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       303 non-null    int64  
 1   sex       303 non-null    int64  
 2   cp        303 non-null    int64  
 3   trtbps    303 non-null    int64  
 4   chol      303 non-null    int64  
 5   fbs       303 non-null    int64  
 6   restecg   303 non-null    int64  
 7   thalachh  303 non-null    int64  
 8   exng      303 non-null    int64  
 9   oldpeak   303 non-null    float64
 10  slp       303 non-null    int64  
 11  caa       303 non-null    int64  
 12  thall     303 non-null    int64  
 13  output    303 non-null    int64  
dtypes: float64(1), int64(13)
"""

sns.heatmap(df.corr(),annot = True,fmt = '.1f')

cp = df["cp"]
cp = pd.get_dummies(cp,drop_first=True)
cp = cp.rename({1:"cp1",2:"cp2",3:"cp3"},axis=1)

"""
     cp1  cp2  cp3
0      0    0    1
1      0    1    0
2      1    0    0
3      1    0    0
4      0    0    0
"""

ecg = df["restecg"]
ecg = pd.get_dummies(ecg,drop_first=True)
ecg = ecg.rename({1:"ecg1",2:"ecg2"},axis=1)

"""
     ecg1  ecg2
0       0     0
1       1     0
2       0     0
3       1     0
4       1     0
"""

slp = df["slp"]
slp = pd.get_dummies(slp,drop_first=True)
slp = slp.rename({1:"slp1",2:"slp2"},axis=1)
"""
     slp1  slp2
0       0     0
1       0     0
2       0     1
3       0     1
4       0     1
"""

thall = df["thall"]
thall = pd.get_dummies(thall,drop_first=True)
thall = thall.rename({1:"thall1",2:"thall",3:"thall3"},axis=1)

"""
    thall1  thall  thall3
0         1      0       0
1         0      1       0
2         0      1       0
3         0      1       0
4         0      1       0
"""

df = df.drop(["cp","restecg","slp","thall"],axis=1)
df = pd.concat([df,cp,ecg,slp,thall],axis=1)

X = df.drop("output",axis=1)
y = df["output"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization

model = Sequential()
model.add(Dense(40,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(30,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(20,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(10,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(1,activation="sigmoid"))
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(optimizer=opt,loss=loss,metrics=["accuracy"])
history = model.fit(x_train,y_train,batch_size=32,epochs=700,validation_data=(x_test,y_test))

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
Epoch 695/700
8/8 [==============================] - 0s 3ms/step - loss: 0.3994 - accuracy: 0.8116 - val_loss: 0.4084 - val_accuracy: 0.8525
Epoch 696/700
8/8 [==============================] - 0s 3ms/step - loss: 0.3797 - accuracy: 0.8448 - val_loss: 0.4086 - val_accuracy: 0.8525
Epoch 697/700
8/8 [==============================] - 0s 2ms/step - loss: 0.3154 - accuracy: 0.8831 - val_loss: 0.4085 - val_accuracy: 0.8525
Epoch 698/700
8/8 [==============================] - 0s 2ms/step - loss: 0.2983 - accuracy: 0.9042 - val_loss: 0.4085 - val_accuracy: 0.8525
Epoch 699/700
8/8 [==============================] - 0s 2ms/step - loss: 0.3238 - accuracy: 0.8755 - val_loss: 0.4089 - val_accuracy: 0.8525
Epoch 700/700
8/8 [==============================] - 0s 2ms/step - loss: 0.3351 - accuracy: 0.8764 - val_loss: 0.4088 - val_accuracy: 0.8525
"""
