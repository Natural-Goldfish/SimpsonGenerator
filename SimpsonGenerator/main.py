import argparse
from training import train
from test import generate

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', '-m', choices= ["test", "train"], required= True, \
        help = "There are two types mode, Test mode is to generate new image using trained model and Train mode is to train your model")
    parser.add_argument('--model_path', default = "data\\models")

    # For training
    parser.add_argument('--epoch', type = int, default = 400)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--learning_rate', '-lr', type = float, default = 0.0002)
    parser.add_argument('--betas', type = float , nargs = 2, default = (0.5, 0.999), \
        help = "Hyperparameters to be used on the optimizer")
    parser.add_argument('--image_path', default = "data\\images")

    # For continuous training
    parser.add_argument('--model_load_flag', action = 'store_true', \
        help = "When you want to keep training your model, set True. If it's True, you must write the name of the model you are going to load")
    parser.add_argument('--generator_load_name', required= False, default = "generator_385_checkpoint.pth",\
        help = "When the 'model_load_flag' is True, This is required to load generator model to train continuously")       
    parser.add_argument('--discriminator_load_name', required= False, default = "temp",\
        help = "When the 'model_load_flag' is True, This is required to load discriminator model to train continuously")

    # For generating
    parser.add_argument('--generating_model_name', required= False, default = 'generator_399_checkpoint.pth')
    parser.add_argument('--image_save_path', default = "data\\generated_images",\
        help = "All of the images which the generator model generates are stored in this directory")
    parser.add_argument('--generate_numbers', '-n', type = int, default = 1)
    
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else :
        generate(args)
if __name__ == "__main__":
    get_args()
