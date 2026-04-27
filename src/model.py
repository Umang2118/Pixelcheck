import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_model(device):
    """
    Load a pre-trained EfficientNet-B0 model and adapt the final classifier
    for our binary (Real vs Fake) image detection task.
    EfficientNet-B0 is fast, lightweight, and highly accurate.
    """
    # Load the base model with the best available pre-trained weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    
    # Get the number of input features to the final classification layer
    num_ftrs = model.classifier[1].in_features
    
    # Replace the final layer to output exactly 2 classes (Fake=0, Real=1)
    # We add dropout to prevent overfitting on our dataset
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, 2)
    )
    
    # Move the model to the specified device (GPU or CPU)
    model = model.to(device)
    return model