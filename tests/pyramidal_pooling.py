import torch
import torch.nn as nn

import torch.nn.functional as F

class PyramidalPoolingModule(nn.Module):
    def __init__(self, input_channels,pool_sizes,output_channels):
        super(PyramidalPoolingModule, self).__init__()
        self.pooling_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d((pool_size, pool_size))  # AdaptiveAvgPool2d with variable pool sizes
            for pool_size in pool_sizes
        ])
        if isinstance(output_channels, int):
            self.output_channels = [output_channels] * len(self.pooling_layers)
        else:
            assert len(output_channels) == len(self.pooling_layers), "Length of output_channels must match the number of pooling layers"
            self.output_channels = output_channels
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(input_channels, out_channels, kernel_size=1)  # Convolutional layer to adjust channel dimension
            for out_channels in self.output_channels
        ])

    def forward(self, x):
        pooled_features = []
        for pooling_layer, conv_layer in zip(self.pooling_layers, self.conv_layers):
            pooled_feature = pooling_layer(x)
            pooled_feature = F.relu(conv_layer(pooled_feature))  # Apply convolution to adjust channel dimension
            pooled_feature = F.interpolate(pooled_feature, size=x.size()[2:], mode='bilinear', align_corners=True)  # Resize to original spatial resolution
            pooled_features.append(pooled_feature)
                # Adjust the number of output channels to match the desired final output
        total_output_channels = sum(self.output_channels)
        output_features = torch.cat(pooled_features, dim=1)  # Concatenate pooled features along channel dimension
        output_features = output_features[:, :total_output_channels, :, :]  # Trim to desired output channels
        return output_features
