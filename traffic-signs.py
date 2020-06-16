# -*- coding: utf-8 -*-
"""
Created on Tue May 26 04:57:07 2020

@author: kingslayer
"""

#importing libararies and dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

with open(r"traffic-signs-data\train.p",mode="rb") as training_data:
    train=pickle.load(training_data)
with open(r"traffic-signs-data\test.p",mode="rb") as test_data:
    test=pickle.load(test_data)
with open(r"traffic-signs-data\valid.p",mode="rb") as valid_data:
    valid=pickle.load(valid_data)
    
X_train,y_train=train["features"],train["labels"]
X_val,y_val=valid["features"],valid["labels"]
X_test,y_test=test["features"],test["labels"]

#Data Visualisation
W_grid=15
L_grid=15
fig,axis=plt.subplots(W_grid,L_grid,figsize=(20,20))
axis=axis.ravel()
n_training=len(X_train)


for i in np.arange(0,L_grid*W_grid):
    index=np.random.randint(0,n_training)
    axis[i].imshow(X_train[index])
    axis[i].set_title(y_train[index])
    axis[i].axis("off")
    
#color to gray scale
    
X_train=np.sum(X_train/3,axis=3,keepdims=True)
X_test=np.sum(X_test/3,axis=3,keepdims=True)
X_val=np.sum(X_val/3,axis=3,keepdims=True)
    
X_train=X_train/255
X_test=X_test/255
X_val=X_val/255

plt.imshow(X_train[1000].squeeze(),cmap="gray")


#Model 
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten

cnn=Sequential()

cnn.add(Conv2D(64,(3,3),activation="relu",input_shape=(32,32,1)))
cnn.add(MaxPooling2D(2,2))
cnn.add(Dropout(0.2))

cnn.add(Conv2D(128,(3,3),activation="relu"))
cnn.add(MaxPooling2D(2,2))
cnn.add(Dropout(0.2))

cnn.add(Flatten())

cnn.add(Dense(units=512,activation="relu"))
cnn.add(Dropout(0.2))
cnn.add(Dense(units=512,activation="relu"))
cnn.add(Dense(units=43,activation="sigmoid"))

cnn.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
history=cnn.fit(X_train,y_train,batch_size=100,epochs=10,validation_data=(X_val,y_val))

history.history.keys()

plt.figure(figsize=(10,10))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train loss","Validation loss"])

plt.figure(figsize=(10,10))
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train Accuracy","Validation Accuracy"])


y_pred=cnn.predict_classes(X_test)

accuracy=cnn.evaluate(X_test,y_test)
accuracy=accuracy[1]

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

plt.figure(figsize=(15,15))
sns.heatmap(cm,annot=True)

#Visualisation
L_grid=15
W_grid=15
fig,axis=plt.subplots(L_grid,W_grid,figsize=(20,20))
axis=axis.ravel()
data_len=len(X_test)

for i in np.arange(0,L_grid*W_grid):
    index=np.random.randint(data_len)
    axis[i].imshow(X_test[index].squeeze())
    axis[i].set_title(f"{y_test[index]}\n{y_pred[index]}")
    axis[i].axis("off")

plt.subplots_adjust(top=2)