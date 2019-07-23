from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.initializers import VarianceScaling

from parameters import RESIZE_DIMENSION


def build_model():
    model = Sequential()

    # First convolutional layer
    model.add(Conv2D(20, 8, strides=(4, 4),
                          padding='valid',
                          activation='relu',
                          data_format='channels_last',
                          input_shape=(RESIZE_DIMENSION[1], RESIZE_DIMENSION[0], 1),
                          kernel_initializer=VarianceScaling(scale=2.0),))
    model.add(Dropout(0.5))

    # Second convolutional layer
    model.add(Conv2D(25, 5, strides=(4, 4),
                     padding='valid',
                     activation='relu',
                     data_format='channels_last',
                     kernel_initializer=VarianceScaling(scale=2.0), ))
    model.add(Dropout(0.5))

    # Third convolutional layer
    model.add(Conv2D(30, 4, strides=(2, 2),
                     padding='valid',
                     activation='relu',
                     data_format='channels_last',
                     kernel_initializer=VarianceScaling(scale=2.0), ))
    model.add(Dropout(0.5))

    # Fourth convolutional layer
    model.add(Conv2D(40, 4, strides=(2, 2),
                          padding='valid',
                          activation='relu',
                          kernel_initializer=VarianceScaling(scale=2.0),))
    model.add(Dropout(0.5))

    # Fifth convolutional layer
    model.add(Conv2D(64, 3, strides=(1, 1),
                          padding='valid',
                          activation='relu',
                          kernel_initializer=VarianceScaling(scale=2.0),))
    model.add(Dropout(0.5))

    # Flatten the convolution output
    model.add(Flatten())

    # dense layer
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model
