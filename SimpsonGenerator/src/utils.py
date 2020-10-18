import numpy as np
import cv2
import os

def save_images(images, save_path, epoch):
    images = np.transpose(images, axes = (0, 2, 3, 1))
    for i in range(len(images)):
        images[i] = images[i]*255
        save_name = "{}_{}.jpg".format(epoch, i)
        cv2.imwrite(os.path.join(save_path, save_name), images[i])

