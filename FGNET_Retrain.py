# -*- coding: utf-8 -*-

from keras.models import load_model
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import os
import cv2
import torch
from torchvision.transforms import ToTensor
from PIL import Image, ImageEnhance
from torch.autograd import Variable
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import h5py
import glob
from numpy import expand_dims
image_list = []

batch_size = 1  # Numero de muestras para cada batch (grupo de entrada)


def load_test():
    X_test = []
    images_names = []
    image_path = "/home/fariborz/PycharmProjects/New_Projects/FaceQnet-master/src/Test_samples/JPG/12"
    print('Read test images')
    # for f in [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]:
    # carpeta= os.path.join(image_path, f)
    # print(carpeta)
    # for imagen in [imagen for imagen in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, imagen))]:
    # imagenes = os.path.join(carpeta, imagen)
    # img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (224, 224))
    # X_test.append(img)
    # images_names.append(imagenes)

    for imagen in [imagen for imagen in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, imagen))]:
        imagenes = os.path.join(image_path, imagen)
        print(imagenes)
        img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (224, 224))
        X_test.append(img)
        images_names.append(imagenes)
    return X_test, images_names


def read_and_normalize_test_data():
    test_data, images_names = load_test()
    test_data = np.array(test_data, copy=False, dtype=np.float32)

    return test_data, images_names


# # get the face embedding for one face
# def get_embedding(model, face_pixels):
#     # scale pixel values
#     face_pixels = face_pixels.astype('float32')
#     # standardize pixel values across channels (global)
#     mean, std = face_pixels.mean(), face_pixels.std()
#     face_pixels = (face_pixels - mean) / std
#     # transform face into one sample
#     samples = expand_dims(face_pixels, axis=0)
#     # make prediction to get embedding
#     yhat = model.predict(samples)
#     return yhat[0]from torchvision.transforms import ToTensor
#
#
# # Loading the pretrained model
# #model = load_model('Fariborz.h5')
# model = load_model('facenet_keras.h5')
# # summarize input and output shape
# print(model.inputs)
# print(model.outputs)

# See the details (layers) of FaceQnet
# print(model.summary())

# Loading the test data
test_data, images_names = read_and_normalize_test_data()
#im = Image.open('/home/fariborz/PycharmProjects/New_Projects/FaceQnet-master/src/Test_samples/JPG/12/0001_01.jpg')

img = cv2.imread('/home/fariborz/PycharmProjects/New_Projects/FaceQnet-master/src/Test_samples/JPG/12/0001_01.jpg')
img = cv2.resize(img, (160, 160))
resnet = InceptionResnetV1(pretrained='vggface2').eval()
img = ToTensor()(img).unsqueeze(0) # unsqueeze to add artificial first dimension
image = Variable(img)

img_embedding = resnet(image)
print(img_embedding)
h=0

# Feature=get_embedding(model, x)
# print(Feature)
# if test_data.ndim == 4:
# axis = (1, 2, 3)
# size = test_data[0].size
# elif test_data.ndim == 3:
# axis = (0, 1, 2)
# size = test_data.size
# else:
# raise ValueError('Dimension should be 3 or 4')

# mean = np.mean(test_data, axis=axis, keepdims=True)
# std = np.std(test_data, axis=axis, keepdims=True)
# std_adj = np.maximum(std, 1.0/np.sqrt(size))
# y = (test_data - mean) / std_adj

y = test_data

# Extract quality scores for the samples
m = 0.7
s = 0.5
score = model.predict(y, batch_size=batch_size, verbose=1)

# score = 0.5*np.tanh(((score-m)/s) + 1)
# predictions = -score + 1; #Convertimos el score de distancia en similitud
predictions = score
# Guardamos los scores para cada clase en la prediccion de scores
