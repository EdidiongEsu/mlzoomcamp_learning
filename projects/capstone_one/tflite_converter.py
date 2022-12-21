import numpy as np

import tensorflow.lite as tflite
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import load_img

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.applications.xception import decode_predictions

from keras_preprocessing.image import ImageDataGenerator
from keras_image_helper import create_preprocessor

model = keras.models.load_model('xception_v4_1_09_0.934.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('car-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)



