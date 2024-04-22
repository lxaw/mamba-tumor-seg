import torch 

def calculate_iou(pred_mask, true_mask):

    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()

    iou = intersection / union if union != 0 else 0.0

    # Check if iou is a tensor
    if torch.is_tensor(iou):
        iou = iou.item()  # Convert tensor to Python float

    return iou