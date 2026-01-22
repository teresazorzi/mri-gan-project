import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    """
    Custom weights initialization called on Generator and Discriminator.
    Initializes Conv layers with Normal(0, 0.02) and BatchNorm/InstanceNorm with Normal(1, 0.02).
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('InstanceNorm') != -1:
        if m.weight is not None: 
            nn.init.normal_(m.weight.data, 1.0, 0.02)

class CPUOptimizedGenerator3D(nn.Module):
    """
    3D Generator Network.
    Takes a latent vector (z) and a class label, outputs a 3D MRI volume.
    Designed to be lightweight for CPU/Entry-level GPU execution.
    """
    def __init__(self, latent_dim=64, num_classes=3, ngf=32, target_shape=(64, 64, 64)):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_shape = target_shape
        
        # Label Embedding
        self.label_embedding = nn.Embedding(num_classes, latent_dim // 2)
        input_dim = latent_dim + latent_dim // 2
        
        self.initial = nn.Sequential(
            nn.ConvTranspose3d(input_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.InstanceNorm3d(ngf * 8, affine=True),
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(ngf * 4, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(ngf * 2, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(ngf, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh() # Outputs values in [-1, 1]
        )

    def forward(self, z, labels):
        emb = self.label_embedding(labels)
        # Concatenate noise and label embedding
        x = torch.cat([z, emb], dim=1).view(-1, self.latent_dim + self.latent_dim // 2, 1, 1, 1)
        x = self.initial(x)
        x = self.main(x)
        return F.interpolate(x, size=self.target_shape, mode='trilinear')

class CPUOptimizedDiscriminator3D(nn.Module):
    """
    3D Discriminator Network (Critic).
    Takes a 3D volume and a class label, outputs a scalar validity score (WGAN logic).
    """
    def __init__(self, num_classes=3, ndf=32, input_shape=(64, 64, 64)):
        super().__init__()
        self.input_shape = input_shape
        
        # Label Embedding mapped to image dimensions
        self.label_embedding = nn.Embedding(num_classes, 32)
        self.label_proj = nn.Linear(32, input_shape[0]*input_shape[1]*input_shape[2])
        
        self.main = nn.Sequential(
            nn.Conv3d(2, ndf, 4, 2, 1), # Input channels = 2 (Image + Label channel)
            nn.LeakyReLU(0.2),
            nn.Conv3d(ndf, ndf*2, 4, 2, 1), 
            nn.InstanceNorm3d(ndf*2), 
            nn.LeakyReLU(0.2),
            nn.Conv3d(ndf*2, ndf*4, 4, 2, 1), 
            nn.InstanceNorm3d(ndf*4), 
            nn.LeakyReLU(0.2),
            nn.Conv3d(ndf*4, ndf*8, 4, 2, 1), 
            nn.InstanceNorm3d(ndf*8), 
            nn.LeakyReLU(0.2),
            nn.Conv3d(ndf*8, 1, 4, 1, 0)
        )

    def forward(self, x, labels):
        emb = self.label_embedding(labels)
        # Project embedding to match image size and concatenate as a new channel
        lp = self.label_proj(emb).view(-1, 1, *self.input_shape)
        return self.main(torch.cat([x, lp], dim=1)).view(-1)