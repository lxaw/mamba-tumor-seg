import torch
import torch.nn as nn
import torchvision.models as models



def load_model_and_weights(device):
    model = ResNet18Seg().to(device)

    # Load weights for parallel training
    state_dict = torch.load('resnet18_weights.pth', map_location=device)
    # Remove the 'module.' prefix from keys if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()

    return model


# ResNet18

class ResNet18Seg(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet18Seg, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove the final fully connected layer and avgpool layer
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Upsample to 8x8 spatial size
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample to 16x16 spatial size
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # Upsample to 32x32 spatial size
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1),  # Upsample to 64x64 spatial size (original input size)
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x