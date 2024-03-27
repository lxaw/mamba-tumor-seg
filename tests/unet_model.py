####
# U-Net Model
#
from torch import nn
import torch

class UNet(nn.Module):
   def __init__(self):
       super(UNet, self).__init__()

       # Encoder
       self.encoder1 = nn.Sequential(
           nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
           nn.BatchNorm2d(32),
           nn.ReLU(inplace=True),
           nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
           nn.BatchNorm2d(32),
           nn.ReLU(inplace=True)
       )
       self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
       self.encoder2 = nn.Sequential(
           nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
           nn.BatchNorm2d(64),
           nn.ReLU(inplace=True),
           nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
           nn.BatchNorm2d(64),
           nn.ReLU(inplace=True)
       )
       self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
       self.encoder3 = nn.Sequential(
           nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
           nn.BatchNorm2d(128),
           nn.ReLU(inplace=True),
           nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
           nn.BatchNorm2d(128),
           nn.ReLU(inplace=True)
       )
       self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
       self.encoder4 = nn.Sequential(
           nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
           nn.BatchNorm2d(256),
           nn.ReLU(inplace=True),
           nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
           nn.BatchNorm2d(256),
           nn.ReLU(inplace=True)
       )
       self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


       # Bottleneck
       self.bottleneck = nn.Sequential(
           nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
           nn.BatchNorm2d(512),
           nn.ReLU(inplace=True),
           nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
           nn.BatchNorm2d(512),
           nn.ReLU(inplace=True),
          #  nn.MaxPool2d(kernel_size=2, stride=2)
       )

       # Decoder
       self.upconv4 = nn.Sequential(
           nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2)),
          #  nn.MaxPool2d(kernel_size=2, stride=2)
           )
       self.decoder4 = nn.Sequential(
           nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1)),
           nn.BatchNorm2d(256),
           nn.Tanh(),
           nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
           nn.BatchNorm2d(256),
           nn.Tanh()
       )
       self.upconv3 = nn.Sequential(
           nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2)),
          #  nn.MaxPool2d(kernel_size=2, stride=2)
           )
       self.decoder3 = nn.Sequential(
           nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
           nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
           nn.Tanh(),
           nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
           nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
           nn.Tanh()
           )
       self.upconv2 = nn.Sequential(
           nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2)),
          #  nn.MaxPool2d(kernel_size=2, stride=2)
           )
       self.decoder2 = nn.Sequential(
           nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
           nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
           nn.Tanh(),
           nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
           nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
           nn.Tanh()
           )
       self.upconv1 = nn.Sequential(
           nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2)),
          #  nn.MaxPool2d(kernel_size=2, stride=2)
           )
       self.decoder1 = nn.Sequential(
           nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
           nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
           nn.Tanh(),
           nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
           nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
           nn.Tanh()
           )
       self.conv = nn.Sequential(
           nn.ConvTranspose2d(32, 3, kernel_size=(1, 1), stride=(1, 1)),
           nn.Sigmoid()
           )

   def forward(self, x):
        #Encoder
       encoder1 = self.encoder1(x)
       encoder2 = self.encoder2(self.pool1(encoder1))
       encoder3 = self.encoder3(self.pool2(encoder2))
       encoder4 = self.encoder4(self.pool3(encoder3))

       # Bottleneck
       bottleneck = self.bottleneck(self.pool4(encoder4))

       #Decoder & Connections
       x = self.upconv4(bottleneck)
       x = torch.cat([x, encoder4], dim=1)
       x = self.decoder4(x)

       x = self.upconv3(x)
       x = torch.cat([x, encoder3], dim=1)
       x = self.decoder3(x)

       x = self.upconv2(x)
       x = torch.cat([x, encoder2], dim=1)
       x = self.decoder2(x)

       x = self.upconv1(x)
       x = torch.cat([x, encoder1], dim=1)
       x = self.decoder1(x)

       x = self.conv(x)

       return x #(x >= 0.6).float()
