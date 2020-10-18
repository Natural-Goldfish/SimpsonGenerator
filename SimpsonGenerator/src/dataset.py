from torch.utils.data import Dataset
from src.data_argumentation import *
import os

_IMAGE_PATH = "data\\images"

class SimpsonDataset(Dataset):
    def __init__(self, image_path = _IMAGE_PATH):
        self.image_path = image_path
        self.image_length = len(os.listdir(self.image_path))

    def __len__(self):
        return self.image_length

    def __getitem__(self, idx):
        file_name = "{}.png".format(idx+1)
        image = cv2.imread(os.path.join(self.image_path, file_name))
        transforms = Transforms([Flip(), Resize(), Normalize(), Numpy2Tensor()])
        image = transforms(image)
        return image