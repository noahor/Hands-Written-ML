import keras
from keras.datasets import mnist
from keras.preprocessing.image import load_img, array_to_img
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from skimage import io, color
from PIL import Image
import matplotlib.pyplot as plt
import imageio



#  TensorBoard logger
logger = keras.callbacks.TensorBoard(

    
    log_dir= "c:\logs",
    write_graph=True,
    histogram_freq=5
)

# 1. Define image height and width 
image_height,image_width = 28,28

# 2. Load mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 3. Data preperation for using mnist data
X_train = X_train.reshape(60000, image_height*image_width)
X_test = X_test.reshape(10000, image_height*image_width)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# 4. Create Your Model please note for your first layer you should add the following 
# an input_shape of 784 because of the image sizes -> 28 height * 28 width = 728. 
# model.add(Dense(xxxxx, activation='xxx', input_shape=(784,)))
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(10,activation='softmax'))

# Complie your Model 
model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy', metrics=['accuracy'])   
# Fit  your Model 
history = model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test),callbacks=[logger])
# evaluate  your Model 
test_error_rate = model.evaluate(X_test, y_test)

_,accuracy = test_error_rate
print('Accuracy: %.2f' % (accuracy*100))

# Save your Model 
model.save("HandWritten.h5")

