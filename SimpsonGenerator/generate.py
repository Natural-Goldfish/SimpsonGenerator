from src.network import Z_Generator, Generator
import torch
import os
import cv2

_MODEL_PATH = "data\\models"
_MODEL_NAME = "generator_{}_checkpoint.pth".format(epoch)

def new_image():
    latent_space = Z_Generator()
    generator = Generator()

    generator.state_dict(torch.load(os.path.join(_MODEL_PATH, _MODEL_NAME)))
    generator.eval()

    latent_vector = latent_space(1)
    new_image = generator(latent_vector)
    new_image = new_image.squeeze(0).permute(1, 2, 0).numpy()

    cv2.imshow("Generated_img", new_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    new_image()