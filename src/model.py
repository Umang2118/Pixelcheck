import torch
import torch.nn as nn
from transformers import CvtForImageClassification

class CustomClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 256)
        self.norm1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.BatchNorm1d(128)
        self.fc_out = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = nn.functional.relu(x)
        x = self.fc_out(x)
        return x

def get_model(device):
    """
    Load a pre-trained CvT-13 model and adapt the final classifier
    for our binary (Real vs Fake) image detection task using the
    custom multi-layer classification head.
    """
    model = CvtForImageClassification.from_pretrained(
        'microsoft/cvt-13',
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.classifier = CustomClassifier()
    model = model.to(device)
    return model