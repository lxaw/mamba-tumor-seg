import torch
import torch.nn as nn

from mamba_model import UMambaBot

from torch.utils.data import DataLoader
from brain_mri_dataset import BrainMRIDatasetBuilder,BrainMRIDataset

from transforms import BrainMRITransforms
import json

from metrics import calculate_iou, dice_coefficient


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

# Creating the MRI Dataset
# We only care about test dataset
builder = BrainMRIDatasetBuilder("../datasets/tumor_segs")
df = builder.create_df()
train_df, val_df, test_df = builder.split_df(df)

# transformation
transform_ = BrainMRITransforms()

# dataset
test_dataset = BrainMRIDataset(test_df,transform=transform_)

# batch 
batch_size = 64

# Data Loaders
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)


# Function to evaluate model and store results
def evaluate_model(model, dataloader):
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images.to(device).float()).cpu()

            # Convert logits to predictions (assuming output is softmax)
            predictions = torch.argmax(outputs, dim=1)

            # Calculate scores for each batch
            for i in range(len(images)):
                dice = dice_coefficient(predictions[i], masks[i])
                iou = calculate_iou(predictions[i], masks[i])
                dice_scores.append(dice)
                iou_scores.append(iou)

    avg_dice_score = sum(dice_scores) / len(dice_scores)
    avg_iou_score = sum(iou_scores) / len(iou_scores)

    return avg_dice_score, avg_iou_score

# Dictionary to store evaluation results
evaluation_results = {}

# Define model parameters to evaluate
weights = "mamba_pipeline_py_weights.pth"
model_parameters = [
    {"weights": "mamba_weights_1.pth"},
]

# Iterate over model parameters
for params in model_parameters:
    # Load model with specified parameters
    model.load_state_dict(torch.load(params["weights"]))
    model.eval()

    # Evaluate model
    avg_dice_score, avg_iou_score = evaluate_model(model, test_dataloader)

    # Store evaluation results
    evaluation_results[str(params)] = {"average_dice_score": avg_dice_score, "average_iou_score": avg_iou_score}

# Save evaluation results to JSON file
output_file = "evaluation_results.json"
with open(output_file, "w") as f:
    json.dump(evaluation_results, f)

print("Evaluation results saved to:", output_file)