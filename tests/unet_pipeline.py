################################
#
# Clay Crews, Lex Whalen
# U-Net
# 
# Explanation: 
# Testing with just basic U-Net
#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys

from unet_model import UNet
from dice_loss import DiceLoss
import torch.optim as optim
from brain_mri_dataset import BrainMRIDatasetBuilder,BrainMRIDataset

from transforms import BrainMRITransforms

from metrics import calculate_iou,calculate_dice

import json
import time


OUT_JSON_NAME = "unet_results.json"
OUT_PARAM_PREFIX = "unet_params"


# for cuda parallelization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0,1]

# Check the total number of CUDA devices
num_devices = torch.cuda.device_count()
print("Number of CUDA devices:", num_devices)

# Get the name of each CUDA device
for i in range(num_devices):
    device_name = torch.cuda.get_device_name(i)
    print(f"Device {i}: {device_name}")

# Number parameters
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f'Total parameters: {total_params}')
# print(f'Trainable parameters: {trainable_params}')

learning_rate = 0.0003
epochs = 100

train_loss = []
val_loss = []
trainIOU = []
valIOU = []

builder = BrainMRIDatasetBuilder("../datasets/tumor_segs")
df = builder.create_df()
train_df, val_df, test_df = builder.split_df(df)

transform_ = BrainMRITransforms()

train_data = BrainMRIDataset(train_df, transform = transform_ ,  mask_transform= transform_)
val_data = BrainMRIDataset(val_df, transform = transform_ ,  mask_transform= transform_)
test_data = BrainMRIDataset(test_df, transform = transform_ ,  mask_transform= transform_)

# batch
batch_size = 128

# Number of runs of the models
num_runs = 3

###
# Train the model
def train_model(model,train_dataloader,val_dataloader,train_iter):
    """
    Train iter means what number of training are we on
    We train each model about 3 times
    """
    res = {
        'train_loss':[],
        'val_loss':[],
        'train_dice':[],
        'val_dice':[],
        'train_iou':[],
        'val_iou':[],
        'time':None
    }
    start_time=time.time()
    for epoch in range(epochs):
        total_train_loss = 0.0
        total_train_iou = 0.0
        total_train_dice = 0.0

        total_val_loss = 0.0
        total_val_iou = 0.0
        total_val_dice = 0.0

        model.train()

        for img,label in train_dataloader:
            img,label = img.to(device).float(),label.to(device).float()
            optimizer.zero_grad()
            pred = model(img)

            loss = criterion(pred,label)
            total_train_loss += loss.item()

            iou = calculate_iou(pred,label)
            total_train_iou += iou

            dice = calculate_dice(pred,label)
            total_train_dice +=dice

            loss.backward()
            optimizer.step()

        res['train_dice'].append(total_train_dice/len(train_dataloader))
        res['train_iou'].append(total_train_iou/len(train_dataloader))
        res['train_loss'].append(total_train_loss/len(train_dataloader))
        
        # validation
        model.eval()
        with torch.no_grad():
            for img,label in val_dataloader:
                img,label = img.to(device).float(),label.to(device).float()
                pred = model(img)
                loss = criterion(pred,label)
                iou = calculate_iou(label,pred)
                dice = calculate_dice(label,pred)

                total_val_loss += loss.item()
                total_val_dice += dice
                total_val_iou += iou




            # append data
            res['val_dice'].append(total_val_dice/len(val_dataloader))
            res['val_iou'].append(total_val_iou/len(val_dataloader))
            res['val_loss'].append(total_val_loss/len(val_dataloader))
        if epoch % 10 == 0 and epoch != 0:
            print(f'epoch: {epoch}\nres: {res}')

    end_time = time.time()
    res['time'] = (end_time - start_time)
    
    # save model params
    model_state_dict = model.state_dict()
    file_path = f'{OUT_PARAM_PREFIX}_{train_iter}.pth'
    torch.save(model_state_dict,file_path)

    print(f'result: {res}')

    return res


# run and record data
for run in range(num_runs):
    # clear gpu usage each run
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    print(f"Training run {run}")
    # reinitialize the model
    model = UNet()
    # move to gpu
    model = nn.DataParallel(model,device_ids=device_ids).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate)
    criterion = DiceLoss()

    ###
    # Data loaders
    train_dataloader = DataLoader(train_data, batch_size = batch_size , shuffle = True)
    val_dataloader = DataLoader(val_data, batch_size = batch_size , shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = batch_size , shuffle = True)

    results = []

    res = train_model(model=model,train_dataloader=train_dataloader,val_dataloader=val_dataloader,train_iter=run)

    results.append(res)

    with open(OUT_JSON_NAME,"w") as f:
        json.dump(results,f)

    print("Training results saved to: ",OUT_JSON_NAME)