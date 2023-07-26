import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class AE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, args):
        super(AE, self).__init__()
        self.n_z = args['n_z']
        self.nc = args['n_channel']
        
        im_chan = self.nc
        n_z = self.n_z
        hidden_dim = 64

        self.n_class = args['n_class']

        self.encoder = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, n_z, final_layer=True), # 100 channels output
            nn.Flatten()
        )

        # Build the neural network
        self.decoder = nn.Sequential(
            View((-1, n_z, 1, 1)), # unsqueeze the z to generate image
            self.make_gen_block(n_z, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )
        
        self.classifier = nn.Sequential(
            # a simple linear regression model for classification, for the use of SCDL
            nn.Linear(n_z, self.n_class)
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):

      if not final_layer:
          return nn.Sequential(
              nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
              nn.BatchNorm2d(output_channels),
              nn.ReLU(inplace=True),
          )
      else:
          return nn.Sequential(
              nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
              nn.Tanh(),
          )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):

      if not final_layer:
          return nn.Sequential(
              nn.Conv2d(input_channels, output_channels, kernel_size, stride),
              nn.BatchNorm2d(output_channels),
              nn.LeakyReLU(0.2, inplace=True),
          )
      else:
          return nn.Sequential(
              nn.Conv2d(input_channels, output_channels, kernel_size, stride),
          )

    def forward(self, x):
      z = self.encoder(x)
      y = self.classifier(z)
      x_recon = self.decoder(z)

      return x_recon, z, y

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.n_z = args['n_z']
        self.nc = args['n_channel']
        nc = self.nc
        n_z = self.n_z
        self.n_class = args['n_class']
        self.generator = nn.Sequential(
            nn.Linear(n_z, 32*8*7*7),                           # B, nc*4*7*7
            View((-1, 32*8, 7, 7)),                              #B, nc*4, 7, 7  
            nn.ConvTranspose2d(in_channels=32*8,out_channels=32*4, kernel_size=1, stride=1, padding=0, bias=True),   #B, nc*2, 14, 14 
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(in_channels=32*4,out_channels=32*2, kernel_size=2, stride=2, padding=0, bias=True),    # B,  nc, 28, 28
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(in_channels=32*2,out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),   #B, nc*2, 14, 14 
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(in_channels=32,out_channels=1, kernel_size=2, stride=2, padding=0, bias=True),    # B,  nc, 28, 2
        )

    def forward(self, z):
        x_recon = self.generator(z)
        return x_recon


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.n_z = args['n_z']
        self.nc = args['n_channel']
        nc = self.nc
        n_z = self.n_z
        self.n_class = args['n_class']
        self.discriminator = nn.Sequential(
            nn.Conv2d(nc, 32, 3, stride=2, padding=1, bias=True),  #B, nc*2, 14, 14            
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32*2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32*2, 32*4, 3, stride=2, padding=1, bias=True),  #B, nc*2, 14, 14            
            nn.LeakyReLU(0.3),
            nn.Conv2d(32*4, 32*8, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.3),
            nn.Flatten(),                       # B, nc*4*7*7    
            nn.Linear(32*8*7*7, n_z)                    # B, z_dim
        )
        self.last_layer = nn.Linear(n_z, self.n_class +1)

    def forward(self, x):
        z = self.discriminator(x)
        validity = self.last_layer(z)
        return validity
        
# the Classifier in the GAN training
class GAN_Classifier(nn.Module):
    def __init__(self, args, num_c):
        super(GAN_Classifier, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        self.num_class = num_c 

        hidden_dim = 64
        im_chan = self.n_channel
        num_class = 10
        n_z = self.n_z

        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, n_z, final_layer=True), 
            nn.Linear(n_z, num_class + 1)# (n_class + 1) channels output
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):

      if not final_layer:
          return nn.Sequential(
              nn.Conv2d(input_channels, output_channels, kernel_size, stride),
              nn.BatchNorm2d(output_channels),
              nn.LeakyReLU(0.2, inplace=True),
          )
      else:
          return nn.Sequential(
              nn.Conv2d(input_channels, output_channels, kernel_size, stride),
              nn.Flatten()
          )

    def forward(self, x):
        disc_pred = self.disc(x)
        
        return disc_pred