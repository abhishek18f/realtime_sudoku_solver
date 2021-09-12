import tensorflow as tf
import numpy as np
import cv2

# load the model
model = tf.keras.models.load_model('./digit_recognizer/mnist.h5')

img = cv2.imread('null.jpg')

def predict(img):
    # preprocess the image
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img , (28,28))
    img = img.reshape((1,28,28,1))
    results = model.predict(img)
    return print(np.argmax(results,axis = 1)[0])

# print(img.shape)
predict(model.summary())









