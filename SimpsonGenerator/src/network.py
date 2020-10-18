import torch

_LATENT_SPACE_CHANNEL = 100
_BATCH_SIZE = 64

class Z_Generator(object):
    def __init__(self, channel= _LATENT_SPACE_CHANNEL):
        self.latent_space_channel = channel

    def __call__(self, batch_size = _BATCH_SIZE):
        return torch.randn(batch_size, self.latent_space_channel, 1, 1)

class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels = _LATENT_SPACE_CHANNEL, out_channels = 1024, kernel_size = 4, stride = 1, padding = 0, bias = False), # output_size (1024, 4, 4)
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace = True),

            torch.nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias = False),                                                                                # output_size (512, 8, 8)
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace = True),

            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),                                                                                 # output_size (256, 16, 16)
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace = True),

            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),                                                                                 # output_size (128, 32, 32)
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace = True),

            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),                                                                                  # output_size (64, 64, 64)
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace = True),

            torch.nn.Conv2d(64, 3, 1, 1, 0, bias = False),                                                                                             # output_size (64, 3, 3)
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(
                                                                                                                                           # input_size  (3, 64, 64)
            torch.nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 1, stride = 1, padding = 0, bias = False),       # output_size (64, 64, 64)
            torch.nn.LeakyReLU(negative_slope = 0.2, inplace = True),

            torch.nn.Conv2d(64, 128, 4, 2, 1, bias = False),                                                                               # output_size (128, 32, 32)
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, True),         

            torch.nn.Conv2d(128, 256, 4, 2, 1, bias = False),                                                                              # output_size (256, 16, 16)
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, True),

            torch.nn.Conv2d(256, 512, 4, 2, 1, bias = False),                                                                              # output_size (512, 8, 8)
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, True),

            torch.nn.Conv2d(512, 1024, 4, 2, 1, bias = False),                                                                             # output_size (1024, 4, 4)
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2, True),

            torch.nn.Conv2d(1024, 1, 4, 1, 0, bias = False),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.002)
    
    elif classname.find('Batch') != -1 :
        torch.nn.init.normal_(m.weight.data, 1.0, 0.002)
        torch.nn.init.constant_(m.bias.data, 0)