################################
#
# Clay Crews, Lex Whalen
# Pyramidal U-Mamba
# 
# Explanation: 
# We take the original U-Mamba architecture (see here: https://github.com/bowang-lab/U-Mamba) and add
# Pyramidal Pooling 
#
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mamba_model import UMambaBot
from dice_loss import DiceLoss
import torch.optim as optim
from brain_mri_dataset import BrainMRIDatasetBuilder,BrainMRIDataset

from transforms import BrainMRITransforms

from calculate_iou import calculate_iou


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

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters: {total_params}')
print(f'Trainable parameters: {trainable_params}')

learning_rate = 0.0003

criterion = DiceLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate)

epochs = 100

train_loss = []
val_loss = []
trainIOU = []
valIOU = []

import sys

builder = BrainMRIDatasetBuilder("../datasets/tumor_segs")
df = builder.create_df()
train_df, val_df, test_df = builder.split_df(df)

transform_ = BrainMRITransforms()

train_data = BrainMRIDataset(train_df, transform = transform_ ,  mask_transform= transform_)
val_data = BrainMRIDataset(val_df, transform = transform_ ,  mask_transform= transform_)
test_data = BrainMRIDataset(test_df, transform = transform_ ,  mask_transform= transform_)

# batch
batch_size = 64

###
# Data loaders
train_dataloader = DataLoader(train_data, batch_size = batch_size , shuffle = True)
val_dataloader = DataLoader(val_data, batch_size = batch_size , shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = batch_size , shuffle = True)

# Open a file for writing
with open('mamba_pipeline.txt', 'w') as f:
    # Redirect stdout to the file

    for epoch in range(epochs):
        total_train_loss = 0
        total_val_loss = 0
        total_train_iou = 0.0

        # Training mode
        model.train()
        for img, label in train_dataloader:
            img, label = img.to(device).float(), label.to(device).float()
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, label)
            total_train_loss += loss.item()
            iou = calculate_iou(label, pred)
            total_train_iou += iou.item()
            loss.backward()
            optimizer.step()

        train_iou = total_train_iou / len(train_dataloader)
        trainIOU.append(train_iou)
        train_loss.append(total_train_loss / len(train_dataloader))

        # Validation mode
        model.eval()
        total_val_iou = 0.0
        with torch.no_grad():
            for image, label in val_dataloader:
                image, label = image.to(device).float(), label.to(device).float()
                pred = model(image)
                loss = criterion(pred, label)
                total_val_loss += loss.item()
                iou = calculate_iou(label, pred)
                total_val_iou += iou.item()

        val_iou = total_val_iou / len(val_dataloader)
        valIOU.append(val_iou)
        total_val_loss = total_val_loss / len(val_dataloader)
        val_loss.append(total_val_loss)

        # Print and log to the file
        sys.stdout = f
        print('Epoch: {}/{}, Train Loss: {:.4f}, Train IOU: {:.4f}, Val Loss: {:.4f}, Val IOU: {:.4f}'.format(epoch + 1, epochs, train_loss[-1], train_iou, total_val_loss, val_iou))
        sys.stdout = sys.__stdout__
        print('Epoch: {}/{}, Train Loss: {:.4f}, Train IOU: {:.4f}, Val Loss: {:.4f}, Val IOU: {:.4f}'.format(epoch + 1, epochs, train_loss[-1], train_iou, total_val_loss, val_iou))

    # Restore stdout to its original value
    sys.stdout = sys.__stdout__

# Assuming your model is named 'model' and you want to save its state_dict
model_state_dict = model.state_dict()

# Specify the file path where you want to save the weights
file_path = 'mamba_pipeline_py_weights.pth'

# Save the model state_dict to the specified file
torch.save(model_state_dict, file_path)