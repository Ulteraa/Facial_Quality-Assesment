from builtins import print

from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import cv2
import pathlib
from numpy import savez_compressed
import numpy as np
import os
from PIL import Image
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

model = load_model('model/facenet_keras.h5')
print('Loaded Model')
X_train=list()
Y_train=list()
file1 = open('/media/fariborz/d35256bc-623b-42e1-b10d-48071cf615af/Arragnged/score.text', 'r')
Lines = file1.readlines()
count=0
for line in Lines:
    currentline = line.split(",")
    path0 = pathlib.PurePath(currentline[0])
    #print(path0)
    st='/media/fariborz/d35256bc-623b-42e1-b10d-48071cf615af/'+ str(path0)
    img=cv2.imread(st)
    img=cv2.resize(img,(160,160))
    embedding=get_embedding(model,img)
    X_train.append(embedding)
    Y_train.append(float(currentline[1]))
    count=count+1
    print(count)
X_train = asarray(X_train)
Y_train = asarray(Y_train)
savez_compressed('FaceNet_Representation.npz', X_train, Y_train)
model_mine = Sequential()
model_mine.add(Dense(128, input_dim=128, kernel_initializer='normal', activation='relu'))
model_mine.add(Dense(128, kernel_initializer='normal', activation='relu'))
model_mine.add(Dense(1, kernel_initializer='normal'))
# Compile model
model_mine.compile(loss='mean_squared_error', optimizer='adam')
# print(X_train.shape)
# print(Y_train.shape)
history = model_mine.fit(X_train, Y_train, epochs=1000, batch_size=256)
model_mine.save("model_mine.h5")
#--------------------------------------------------------------------------------------------------
#prediction
# for line in Lines:
#     currentline = line.split(",")
#     path0 = pathlib.PurePath(currentline[0])
#     #print(path0)
#     st='/media/fariborz/d35256bc-623b-42e1-b10d-48071cf615af/'+ str(path0)
#     img=cv2.imread(st)
#     img=cv2.resize(img,(160,160))
#     embedding=get_embedding(model,img)
#     e=asarray([embedding])
#     L=model_mine.predict(e)
#     print(L)
# stop=1
