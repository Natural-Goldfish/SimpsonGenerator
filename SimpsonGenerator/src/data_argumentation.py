import random
import torch
import cv2
import numpy as np

_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)
_WIDTH = 64
_HEIGHT = 64

class Transforms(object):
    def __init__(self, functions):
        self.function_list = functions

    def __call__(self, image):
        for function in self.function_list:
            image = function(image)
        return image

class Resize(object):
    def __init__(self, width = _WIDTH, height = _HEIGHT):
        super().__init__()
        self.width = width
        self.height = height

    def __call__(self, image):
        image = cv2.resize(image, (self.width, self.height))
        return image

class Normalize(object):
    def __init__(self, mean = _MEAN, std = _STD):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = image/255
        for i in range(3):
            image[i] = (image[i] - self.mean[i])/self.std[i]
        return image
        
class Numpy2Tensor(object):
    def __init__(self):
        super().__init__()

    def __call__(self, image):
        image = torch.tensor(np.transpose(image, axes = (2, 0, 1)), dtype = torch.float32)
        return image

class Flip(object):
    def __init__(self):
        super().__init__()
        self.direction = [0, 1]
    def __call__(self, image):
        if random.choice(self.direction) == 0 :
            image = cv2.flip(image, 1)       # Left/Right
        return image