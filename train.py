from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

from model import build_model
from parameters import BATCH_SIZE
from pre_processing import get_preprocessed_data


def train():
    pre_processed_data = get_preprocessed_data()
    classifier = build_model()

    # shuffling data
    perm = np.arange(pre_processed_data['data'].shape[0])
    np.random.shuffle(perm)
    pre_processed_data['data'] = pre_processed_data['data'][perm]
    pre_processed_data['target'] = pre_processed_data['target'][perm]

    datagen = ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.1)
    train_generator = datagen.flow(
        pre_processed_data['data'],
        pre_processed_data['target'],
        batch_size=BATCH_SIZE,
        subset='training',
    )
    validation_generator = datagen.flow(
        pre_processed_data['data'],
        pre_processed_data['target'],
        batch_size=BATCH_SIZE,
        subset='validation',
    )

    classifier.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    classifier.fit_generator(train_generator,
                             validation_data=validation_generator,
                             epochs=64)

if __name__ == '__main__':
    train()
