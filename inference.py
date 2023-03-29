# Helper packages
import os
import glob
import time
import numpy as np
from PIL import Image
from helpers import *
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
Path.ls = lambda x: list(x.iterdir())

# AI packages
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------- Model Functions ----------------------------------------- #
class Generator(nn.Module):
    def __init__(self, lr_G=2e-4, dropout=False):
        super(Generator, self).__init__()
        
        # Down sampling
        self.layer1_down = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False)  # 128x128
        self.layer2_down = self._make_layer_down(64, 128, 4, 2, 1, False)                    # 64x64
        self.layer3_down = self._make_layer_down(128, 256, 4, 2, 1, False)                   # 32x32
        self.layer4_down = self._make_layer_down(256, 512, 4, 2, 1, False)                   # 16x16
        self.layer5_down = self._make_layer_down(512, 512, 4, 2, 1, False)                   # 8x8
        self.layer6_down = self._make_layer_down(512, 512, 4, 2, 1, False)                   # 4x4
        self.layer7_down = self._make_layer_down(512, 512, 4, 2, 1, False)                   # 2x2
        
        # Connector
        self.layer8_middle = nn.Sequential(nn.LeakyReLU(0.2, True),
                                           nn.Conv2d(512, 512, 4, 2, 1, bias=False),         # 1x1
                                           nn.ReLU(True),
                                           nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False),# 2x2
                                           nn.BatchNorm2d(512))
        
        # Up sampling
        self.layer9_up = self._make_layer_up(1024, 512, 4, 2, 1, False, True)              # 4x4
        self.layer10_up = self._make_layer_up(1024, 512, 4, 2, 1, False, True)             # 8x8
        self.layer11_up = self._make_layer_up(1024, 512, 4, 2, 1, False, True)             # 16x16
        self.layer12_up = self._make_layer_up(1024, 256, 4, 2, 1, False, False)            # 32x32
        self.layer13_up = self._make_layer_up(512, 128, 4, 2, 1, False, False)             # 64x64
        self.layer14_up = self._make_layer_up(256, 64, 4, 2, 1, False, False)              # 128x128
        
        # Output layer
        self.layer15_up = nn.Sequential(nn.ReLU(True),
                                        nn.ConvTranspose2d(128, 2, 4, 2, 1),               # 256x256
                                        nn.Tanh())
        
        
    def _make_layer_down(self, in_c, out_c, kernel, stride, padding, bias=False):
        layers = []
        # Leaky ReLU
        layers.append(nn.LeakyReLU(0.2, True))
        # 2D convolution layer
        layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernel, 
                                stride=stride, padding=padding, bias=bias))
        # Batch normalization
        layers.append(nn.BatchNorm2d(num_features=out_c))
        return nn.Sequential(*layers)
    
    def _make_layer_up(self, in_c, out_c, kernel, stride, padding, bias=False, dropout=False):
        layers = []
        # Leaky ReLU
        layers.append(nn.ReLU(True))
        # 2D convolution layer
        layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel, 
                                         stride=stride, padding=padding, bias=bias))
        # Batch normalization
        layers.append(nn.BatchNorm2d(num_features=out_c))
        if dropout: layers.append(nn.Dropout())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # store skip connections of U-NET architecture
        skip_connections = []
        
        # Down sampling
        x = self.layer1_down(x)
        skip_connections.append(x)
        x = self.layer2_down(x)
        skip_connections.append(x)
        x = self.layer3_down(x)
        skip_connections.append(x)
        x = self.layer4_down(x)
        skip_connections.append(x)
        x = self.layer5_down(x)
        skip_connections.append(x)
        x = self.layer6_down(x)
        skip_connections.append(x)
        x = self.layer7_down(x)
        skip_connections.append(x)
        
        # bottleneck
        x = self.layer8_middle(x)
        skip_connections = skip_connections[::-1]   # reversing list

        # Up sampling
        x = self.layer9_up(torch.cat([skip_connections.pop(0), x], dim=1))
        x = self.layer10_up(torch.cat([skip_connections.pop(0), x], dim=1))
        x = self.layer11_up(torch.cat([skip_connections.pop(0), x], dim=1))
        x = self.layer12_up(torch.cat([skip_connections.pop(0), x], dim=1))
        x = self.layer13_up(torch.cat([skip_connections.pop(0), x], dim=1))
        x = self.layer14_up(torch.cat([skip_connections.pop(0), x], dim=1))
        x = self.layer15_up(torch.cat([skip_connections.pop(0), x], dim=1))
        
        return x

class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss()
    
    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.layer1 = self._make_layer(3, 64, 4, 2, 1, norm=False, act=True)    # 128x128
        self.layer2 = self._make_layer(64, 128, 4, 2, 1, norm=True, act=True)   # 64x64
        self.layer3 = self._make_layer(128, 256, 4, 2, 1, norm=True, act=True)  # 32x32
        self.layer4 = self._make_layer(256, 512, 4, 1, 1, norm=True, act=True)  # 16x16
        self.layer5 = self._make_layer(512, 1, 4, 1, 1, norm=False, act=False)  # 16X16
        
        
    def _make_layer(self, in_c, out_c, kernel,stride, padding, norm=True, act=True):
        layers = []
        # 2D convolution layer
        layers.append(nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=not norm))
        # Batch normalization
        if norm: layers.append(nn.BatchNorm2d(out_c))
        # Activation function
        if act: layers.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

def init_weights(net, gain=0.02):
    
    # weight initializing function
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            # initialize layer weights
            nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            # initialize layer bias
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
    
    # initialize model weights        
    net.apply(init_func)
    return net

def init_model(model, device):
    # Move model to device and initialize its weights
    model = model.to(device)
    model = init_weights(model)
    return model

class Model(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, beta1=.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        
        # Selecting training device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        
        # Creating Generator
        if net_G is None:
            self.net_G = init_model(Generator(dropout=True), self.device)
        else:
            self.net_G = net_G.to(self.device)
        
        # Creating Discriminator
        self.net_D = init_model(Discriminator(), self.device)
        
        # Initializing GAN loss functions (criterion)
        self.GANcriterion = GANLoss().to(self.device)
        self.L1criterion = nn.L1Loss()
        
        # Initializing optimizers for G an D
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
        
    def set_requires_grad(self, model, requires_grad=True):
        # Turn gradient on for parameters
        for p in model.parameters():
            p.requires_grad = requires_grad
            
    def setup_input(self, data):
        # Split input into correct layers
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        # Generate fake image
        self.fake_color = self.net_G(self.L)
        
    def backward_D(self):
        # Calculate loss for discriminator for optimizer step
        
        # Combine grayscale image "L" with generated colors "a, b"
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        
        # Get discriminator prediction and loss for fake image
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        
        # Get discriminator prediction and loss for real image
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        
        # Calculate Total Loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        
    def backward_G(self):
        # Calculate loss for generator for optimizer step
        
        # Combine grayscale image "L" with generated colors "a, b"
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        
        # Get discriminator prediction and loss for fake image
        fake_preds = self.net_D(fake_image)
        
        # Calculate GAN/L1 loss
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        
        # Calculate Total Loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize(self):
        # Update gradient based on loss values for G & D
        
        # Training Discriminator
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        # Training Generator
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
   
     

# ----------------------------------------- Inference ----------------------------------------- #
def color(model, img_path):
    # load image
    img = Image.open(img_path).convert("RGB")
    img = transforms.Resize((512, 512))(img)
    
    # converting image to L*a*b formate tensors
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")
    img_lab = transforms.ToTensor()(img_lab)
    img_lab = img_lab.unsqueeze(1)
    
    # rescaling images to range from [-1, 1]
    L = img_lab[[0], ...] / 50. - 1. 
    ab = img_lab[[1, 2], ...] / 110.
    data = {'L':L, 'ab':ab}
    
    # running image through model
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    
    # reconstructing the image
    fake_color = model.fake_color.detach()
    L = model.L
    
    fake_img = lab_to_rgb(L, fake_color)
    gray_img = L[0].cpu().numpy().transpose(1,2,0)
    
    # showing image
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(gray_img, cmap='gray')
    plt.subplot(122)
    plt.imshow(fake_img[0])
    plt.show()


def main():
    # loading model
    print("Preparing model... ")
    model = Model()
    model_path = input("Enter model path: ")
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully!\n")
    
    # coloring image
    print("Enter quit to exit")
    img_path = input("Enter image path: ")
    while img_path.lower() != "quit":
        color(model, img_path)
        img_path = input("Enter image path: ")
    
if __name__ == "__main__":
    main()