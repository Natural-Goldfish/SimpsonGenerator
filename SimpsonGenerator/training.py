from src.network import *
from src.dataset import *
from src.utils import save_images
from torch.utils.data import DataLoader
import torch

_CUDA_FLAG = torch.cuda.is_available()

def train(args):
    dataset = SimpsonDataset(args.image_path)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

    latent_space = Z_Generator()
    generator = Generator()
    discriminator = Discriminator()

    # Initalize model's parameters
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Load models
    if args.model_load_flag :
        generator.load_state_dict(torch.load(os.path.join(args.model_path, args.generator_load_name)))
        discriminator.load_state_dict(torch.load(os.path.join(args.model_path, args.discriminator_load_name)))

    # Use GPU, if it's available
    if _CUDA_FLAG :
        generator.cuda()
        discriminator.cuda()
    
    # Loss function
    g_criterion = torch.nn.BCELoss()
    d_criterion = torch.nn.BCELoss()

    # Optimizer
    g_optimizer = torch.optim.Adam(generator.parameters(), lr = args.learning_rate, betas = args.betas)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr = args.learning_rate, betas = args.betas)

    for cur_epoch in range(args.epoch):
        # Train
        generator.train()
        discriminator.train()
        for cur_batch_num, images in enumerate(dataloader):
            if _CUDA_FLAG : images = images.cuda()
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            # Make label for discriminator about real images
            d_toutput = discriminator(images).view(-1)
            d_tlabel = torch.ones_like(d_toutput)
            if _CUDA_FLAG : d_tlabel = d_tlabel.cuda()

            # Calculate loss about real iamges
            d_tloss = d_criterion(d_toutput, d_tlabel)
            d_tloss.backward()

            # Generate fake images from latent space
            latent_vectors = latent_space(len(images))
            if _CUDA_FLAG : latent_vectors = latent_vectors.cuda()
            fake_images = generator(latent_vectors)

            # Make label for discriminator about fake images
            d_foutput = discriminator(fake_images.detach()).view(-1)
            d_flabel = torch.zeros_like(d_foutput)
            if _CUDA_FLAG : d_flabel = d_flabel.cuda()

            # Calculate loss about fake iamges
            d_floss = d_criterion(d_foutput, d_flabel)
            d_floss.backward()

            # Update discriminator's parameters
            d_total_loss = (d_tloss + d_floss)/2
            if cur_epoch < 50 :
                if cur_batch_num % 2 == 0 : d_optimizer.step()
            else :
                d_optimizer.step()

            # Make label for generator
            g_output = discriminator(fake_images).view(-1)
            g_label = torch.ones_like(g_output)
            if _CUDA_FLAG : g_label = g_label.cuda()

            # Update generator's parameters
            g_loss = g_criterion(g_output, g_label)
            g_loss.backward()
            g_optimizer.step()

            print("EPOCH {}/{} Iter {}/{} D TLoss {:.6f} FLoss {:.6f} TotalLoss {:.6f} G TotalLoss {:.6f}".format(\
                cur_epoch, args.epoch, cur_batch_num+1, len(dataloader), d_tloss, d_floss, d_total_loss, g_loss))
        if cur_epoch % 30 == 29 :
            with torch.no_grad():
                generator.eval()
                # Save several images which are generated from generator model
                generator.cpu()
                latent_vectors = latent_space(20)
                test_images = generator(latent_vectors)
                save_images(test_images.numpy(), args.image_save_path, cur_epoch)

                # Save model's parameters    
                generator_save_name = "generator_{}_checkpoint.pth".format(cur_epoch)
                discriminator_save_name = "discriminator_{}_checkpoint.pth".format(cur_epoch)
                torch.save(generator.state_dict(), os.path.join(args.model_path, generator_save_name))
                torch.save(discriminator.state_dict(), os.path.join(args.model_path, discriminator_save_name))
                generator.cuda()
