import torch
import torch.nn as nn

from mamba_model import UMambaBot

weights = "mamba_pipeline_py_weights.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UMambaBot(
    input_channels=3,  # Assuming RGB images with 3 channels
    n_stages=5,
    features_per_stage=(32, 64, 128, 256,512),
    conv_op=nn.Conv2d,  # Assuming 2D convolution
    kernel_sizes=(3, 3, 3, 3, 3),  # Adjusted kernel sizes for 2D convolution
    strides=(1, 2, 2, 2, 2),
    num_classes=3,
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

model.load_state_dict(torch.load(weights))
model.eval()