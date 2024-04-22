import torch
import torch.nn as nn
import torchvision.models as models


def load_model_and_weights(device):
    model = ResNet50Seg().to(device)

    # Load weights for parallel training
    state_dict = torch.load('resnet50_weights.pth', map_location=device)
    # Remove the 'module.' prefix from keys if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()

    return model


# ResNet50

class ResNet50Seg(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet50Seg, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove the final fully connected layer and avgpool layer
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=2, padding=1),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
