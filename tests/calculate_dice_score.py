import torch

def calculate_dice_score(predicted_mask, target_mask):
    # # Flatten predicted and target masks
    # predicted_flat = predicted_mask.view(-1)
    # target_flat = target_mask.view(-1)

    # Calculate intersection
    intersection = torch.sum(predicted_mask * target_mask)

    # Calculate Dice coefficient
    dice = (2. * intersection) / (torch.sum(predicted_mask) + torch.sum(target_mask))

    # Check if dice is a tensor
    if torch.is_tensor(dice):
        dice = dice.item()  # Convert tensor to Python float

    return dice
