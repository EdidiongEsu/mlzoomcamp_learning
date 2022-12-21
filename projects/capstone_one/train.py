# ## Import Packages and Data

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import load_img

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.applications.xception import decode_predictions

from keras_preprocessing.image import ImageDataGenerator

# Variables
input_size = 299
learning_rate = 0.001
size = 1000
droprate = 0.2
no_epochs = 5

# Create functions to be used


def make_model(input_size=input_size, learning_rate=learning_rate, size_inner=size,
               droprate=droprate):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)

    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)

    outputs = keras.layers.Dense(7)(drop)

    model = keras.Model(inputs, outputs)

    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model


# train Vectors
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_ds = train_gen.flow_from_directory(
    'Dataset/train',
    target_size=(input_size, input_size),
    batch_size=32
)

# validation vectors
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory(
    'Dataset/validation',
    target_size=(input_size, input_size),
    batch_size=32,
    shuffle=False
)

model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

model.save_weights('model_v4.h5', save_format='h5')
checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v4_1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


history = model.fit(train_ds, epochs=no_epochs, validation_data=val_ds,
                    callbacks=[checkpoint])

print("The model has been saved")
