# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:39:01 2020

@author: Md Almas Rizwan GARBAGE ANN image classification SOLUTION BUT WORKS
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#TF GPU using 70% gpu VRAM cause i like watching The OFFICE while it trains
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess=tf.Session(config=config)
import cv2
import os
dataset_path=os.getcwd()+"\\dataset"
traindata=dataset_path+"\\training_set"
testdata=dataset_path+"\\test_set"

#Data set making

def makedataset(s,s2):
    X=[]
    listOfFiles = os.listdir(s+"\\"+s2)
    
    for i in range(0,len(listOfFiles)):
        X.append(str(s+"\\"+s2+"\\"+str(listOfFiles[i])))
    return X     
X_train=[]#list of training files
#below just get the list of files path
y_train=np.concatenate((np.ones((4000,),),np.zeros((4000,))))#BEcause the data was in seperate foldes class wise and training/test wise 
y_test=np.concatenate((np.ones((1000,),),np.zeros((1000,)))) #cat=1,dogs=0 as i knew the order, 4000cats then 4000dogs etc
data_train=[]
data_test=[]
X_train=makedataset(traindata,"cats") 
X_train.extend(makedataset(traindata,"dogs") )
X_test=makedataset(testdata,"cats") 
X_test.extend(makedataset(testdata,"dogs") )

#creating the actual data (Should have integrated this in the function, but i am lazy AF)
for imagePath in X_train:
    # load the image, resize the image to be 40X40 pixels (ignoring aspect ratio), 
    # flatten the 40X40x3=4800pixel image into a list, and store the image in the data list
    image = cv2.imread(imagePath)
    if image.shape[0]!=0 and image.shape[1]!=0:
        image = cv2.resize(image, (40, 40)).flatten()
        data_train.append(image)
data_train=np.asarray(data_train)
data_train=np.array(data_train, dtype="float") / 255.0
for imagePath in X_test:
    # load the image, resize the image to be 40X40 pixels (ignoring aspect ratio), 
    # flatten the 40X40x3=4800pixel image into a list, and store the image in the data list
    image = cv2.imread(imagePath)
    if image.shape[0]!=0 and image.shape[1]!=0:
        image = cv2.resize(image, (40, 40)).flatten()
        data_test.append(image)
data_test=np.asarray(data_test)
data_test=np.array(data_test, dtype="float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data_train, y_train, test_size=0.001, random_state=30)# I left extra 8 images for visualising in future :D


#making the ANN model
model = Sequential() 
model.add(Dense(1024, input_shape=(4800,),activation="tanh"))   # input layer 4800as there are 40X40X3=4800 pixels in a flattened input image

model.add(Dense(512, activation="tanh",kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))                         # 2nd hidden layer has 512 nodes
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer=tensorflow.keras.optimizers.Adam(lr=1e-5), metrics=["accuracy"])
H=model.fit(trainX, trainY, batch_size = 32, epochs = 200)
predictions = model.predict(data_test, batch_size=32)
ytest1=y_test.reshape((2000,1))#For confusion matrix cause my list has a shape( 2000,)
pred1=(predictions>0.5)#for confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest1, pred1)
print(model.summary)
loss, acc = model.evaluate(data_test, y_test, verbose = 0)
#Test Accuracy 64% train accuracy 99.97% YIKES # Over Ftting Hence Regularisation
#After l2 regularisation test acc. 65.25 and Train acc. 97.26 still overfitting but that's it
# pred2=model.predict(testX)
# for i in range (0,8):
    # cv2.imshow(cv2.putText(testX[i].reshape((40,40,3)), str(pred2[i]+"%cat "+pred2[i]-1+"%dog"), cv2.FONT_HERSHEY_SIMPLEX ,(255, 0, 0)),"result")