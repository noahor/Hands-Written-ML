from keras.datasets import mnist
from keras.preprocessing.image import load_img, array_to_img
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
from keras.models import load_model
from PIL import Image


import numpy as np
import matplotlib.pyplot as plt
import imageio


# 1. Choose your Picture location 
lina_color= io.imread('C:\\Users\\oron.noah\\OneDrive - Zerto\\DeepLearining Workshop\\HandsWritten_Ex\\7.png')
lina_color = resize(lina_color,(28,28))
image_data = color.rgb2gray(lina_color)


# 2. Load Your Model 
model = load_model("C:\\Users\\oron.noah\\Documents\\MY_ML\\mnistCNN_1.h5")
im2arr = image_data.reshape(1,28,28,1)
im2arr = image_data.reshape(1, 28*28)
# Predicting the Test set results
y_pred = model.predict(im2arr)
print(y_pred)
print(y_pred.argmax())