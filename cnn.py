# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:25:36 2020

@author: Md Almas Rizwan
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras import callbacks
#from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#sess=tf.Session(config=config)
#MODEL STARTS HERE
classifier = Sequential()
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(64,3,3 ,padding="same",activation="relu",))
#classifier.add(BatchNormalization(axis=-1))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#classifier.add(Dropout(0.2)) 
classifier.add(Flatten()) 
classifier.add(Dense(256,activation="relu",kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
classifier.add(Dense(128,activation="relu",kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

#classifier.add(BatchNormalization(axis=-1)) 
#classifier.add(Dropout(0.1)) 
classifier.add(Dense(3,activation="softmax"))
classifier.compile(optimizer='adam',loss='categorical_crossentropy', metrics=["accuracy"])

#DATA PREPROCESSING and training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

classifier.fit_generator(training_set,steps_per_epoch=309,epochs=75,validation_data=test_set,validation_steps=69,callbacks=[callbacks.EarlyStopping(patience=11, verbose=1),callbacks.ReduceLROnPlateau(factor=.5, patience=4, verbose=1)])
classifier.save("CatDogPanda.h5")
#restarting Kernel or restart python/editor to load the saved model===================================================================================================================================================================================
from tensorflow.keras.models import load_model
# load model
model = load_model('CatDogPanda.h5')
# summarize model.
model.summary()
#TEsting on my own images, some screen shots from personal/youtube vids
import numpy as np
from tensorflow.keras.preprocessing import image
#testimg=image.load_img("DOG test.jpg",target_size=(64, 64))
#testimg=image.load_img("Panda test.jpg",target_size=(64, 64))
#testimg=image.load_img("cat.4040.jpg",target_size=(64, 64))
testimg=image.load_img("Annotation 2020-04-05 003724.jpg",target_size=(64, 64))
testimg=image.img_to_array(testimg)
testimg=np.expand_dims(testimg,axis=0)
testimg= testimg/255.0
result1=model.predict(testimg)
#CONCLUSION THIS MODEL STRUGGLES TO IDENTIFY CATS,,Cat dataset is really bad too small pics hence get better dataset or you NEED BIGGER MODEL