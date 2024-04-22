import torch
from torch import nn
from mamba_ssm import Mamba

import os
from glob import glob
import pandas as pd 
import cv2
import torch
import torchvision
from torchvision.models import vgg16_bn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks

from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba


## Custom class imports
#


from pyramidal_pooling import PyramidalPoolingModule

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")





def load_model_and_weights(device):
    model = UMambaBot(
        input_channels=3,  # Assuming RGB images with 3 channels
        n_stages=5,
        features_per_stage=(32, 64, 128, 256,512),
        conv_op=nn.Conv2d,  # Assuming 2D convolution
        kernel_sizes=(3, 3, 3, 3, 3),  # Adjusted kernel sizes for 2D convolution
        strides=(1, 2, 2, 2, 2),
        num_classes=1,
        n_conv_per_stage=(1, 1, 1, 1, 1),
        n_conv_per_stage_decoder=(1, 1, 1, 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,  # Assuming 2D instance normalization
        norm_op_kwargs={},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        # Pyramidal Pooling
        ppm_pool_sizes=(1,2,3,6)
    ).to(device)

    # Load weights for parallel training
    state_dict = torch.load('pp_umamba_weights.pth', map_location=device)
    # Remove the 'module.' prefix from keys if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()



    return model





class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]]):
        """
        This class requires the skip connections from the encoder as input during its forward pass.

        The encoder progresses to the bottleneck, where the decoder takes over. The decoder stages are organized based on computation order, with the initial stage having the lowest resolution. In this stage, the decoder takes the bottleneck features and the lowest-resolution skip connection as inputs.

        Each stage of the decoder comprises three parts:

        1) Convolution transpose to upsample the feature maps from the stage below it (or from the bottleneck in the case of the first stage).
        2) Several convolution blocks to facilitate interaction and merging between the two input sources.
        3) (Optional if `deep_supervision=True`) A segmentation output. Consider enabling upsampling of logits.

        Parameters:
        - `encoder`: The encoder network.
        - `num_classes`: Number of output classes.
        - `n_conv_per_stage`: Number of convolutional blocks per decoder stage.
        """
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedResidualBlocks(
                n_blocks = n_conv_per_stage[s-1],
                conv_op = encoder.conv_op,
                input_channels = 2 * input_features_skip,
                output_channels = input_features_skip,
                kernel_size = encoder.kernel_sizes[-(s + 1)],
                initial_stride = 1,
                conv_bias = encoder.conv_bias,
                norm_op = encoder.norm_op,
                norm_op_kwargs = encoder.norm_op_kwargs,
                dropout_op = encoder.dropout_op,
                dropout_op_kwargs = encoder.dropout_op_kwargs,
                nonlin = encoder.nonlin,
                nonlin_kwargs = encoder.nonlin_kwargs,
            ))

            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        r = seg_outputs[0]
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if s == (len(self.stages) - 1):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output
    
class UMambaBot(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 ppm_pool_sizes=None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        if ppm_pool_sizes:
            self.ppm = PyramidalPoolingModule(input_channels=features_per_stage[-1],pool_sizes=ppm_pool_sizes,output_channels=[64,128,256,64]) # adjust as needed, ensure sum is 512
        else:
            self.ppm = None
        # layer norm
        self.ln = nn.LayerNorm(features_per_stage[-1])
        self.mamba = Mamba(
                        d_model=features_per_stage[-1],
                        d_state=16,  
                        d_conv=4,    
                        expand=2,   
                    )
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder)

    def forward(self, x):
        skips = self.encoder(x)
        ###
        # Pyramidal Pooling
        if self.ppm:
            fused_features = self.ppm(skips[-1])
            # fused_features = skips[-1]
        else:
            fused_features = skips[-1]
            # print(fused_features.shape)
        
        B, C = fused_features.shape[:2] # B = batch,
        n_tokens = fused_features.shape[2:].numel()
        img_dims = fused_features.shape[2:]
        fused_features_flat = fused_features.view(B, C, n_tokens).transpose(-1, -2)
        # this is issue line
        fused_features_flat = self.ln(fused_features_flat)
        out = self.mamba(fused_features_flat)
        out = out.transpose(-1, -2).view(B, C, *img_dims)
        skips[-1] = out
        
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)
