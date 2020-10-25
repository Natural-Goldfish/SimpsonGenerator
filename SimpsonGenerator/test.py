from src.network import Z_Generator, Generator
import torch
import numpy as np
import os
import cv2

def generate(args):
    latent_space = Z_Generator()
    generator = Generator()
    generator.load_state_dict(torch.load(os.path.join(args.model_path, args.generating_model_name)))
    generator.eval()
    latent_vector = latent_space(args.generate_numbers)

    images = generator(latent_vector)
    images = images.detach().numpy()
    images = np.transpose(images, axes = (0, 2, 3, 1))

    for i in range(len(images)):
        images[i] = images[i]*255
        cv2.imwrite(os.path.join(args.image_save_path, "Generated_img{}.jpg".format(i)), images[i])