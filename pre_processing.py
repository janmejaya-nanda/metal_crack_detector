import os
from PIL import Image
from PIL import ImageEnhance

import numpy as np
import cv2
from skimage import color
from skimage import io
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import feature

from parameters import RESIZE_DIMENSION


HEALTHY_FENDER_APRON_PATH = './data/YE358311_Fender_apron_2/YE358311_Fender_apron/YE358311_Healthy'
DEFECTIVE_FENDER_APRON_PATH ='./data/YE358311_Fender_apron_2/YE358311_Fender_apron/YE358311_defects' \
                             '/YE358311_Crack_and_Wrinkle_defect'


def get_all_files():
    healthy_images = os.listdir(HEALTHY_FENDER_APRON_PATH)
    defetced_images = os.listdir(DEFECTIVE_FENDER_APRON_PATH)
    return healthy_images, defetced_images


def pre_processing(data_path):
    img = cv2.imread(data_path)

    # resizing to fix dimension
    resized = cv2.resize(img, RESIZE_DIMENSION, interpolation=cv2.INTER_AREA)

    # Converting to Gray scale and apply differention
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5, scale=2)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5, scale=2)
    plt.imshow(sobelx)
    plt.show()

    return sobelx[..., np.newaxis], sobely[..., np.newaxis]


def get_preprocessed_data():
    healthy_images, defected_images = get_all_files()
    data, target = np.array([]), np.array([])
    for image in healthy_images:
        grad_x, grad_y = pre_processing(data_path=os.path.join(HEALTHY_FENDER_APRON_PATH, image))
        if data.size != 0 and target.size != 0:
            data = np.concatenate((data, grad_x[np.newaxis, ...], grad_y[np.newaxis, ...]), axis=0)
            target = np.append(target, [1, 1])
            # import pdb
            # pdb.set_trace()
        else:
            data = np.concatenate((grad_x[np.newaxis, ...], grad_y[np.newaxis, ...]), axis=0)
            target = np.array([1, 1])

    for image in defected_images:
        grad_x, grad_y = pre_processing(data_path=os.path.join(DEFECTIVE_FENDER_APRON_PATH, image))
        data = np.concatenate((data, grad_x[np.newaxis, ...], grad_y[np.newaxis, ...]), axis=0)
        target = np.append(target, [0, 0])

    return {'data': data, 'target': target}



if __name__ == '__main__':
    get_preprocessed_data()