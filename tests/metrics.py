import torch 

def calculate_iou(pred_mask, true_mask):
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()

    iou = intersection / union if union != 0 else 0.0
    return iou.item()

def calculate_dice(prediction,target):
    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum()
    dice = (2. * intersection) / (union + 1e-8) # add small epsilon to prevent division by 0
    return dice.item()