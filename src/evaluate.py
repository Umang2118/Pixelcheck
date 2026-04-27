import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import glob, os
from model import get_model
from custom_dataset import get_dataset

def evaluate_model(test_data_dir, device, weights_folder):
    test_data = get_dataset(test_data_dir)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    model = get_model(device)
    
    # Load Latest Weights
    weights = glob.glob(os.path.join(weights_folder, '*.pth'))[0]
    model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = outputs.logits.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print(f"Accuracy: {accuracy_score(all_labels, all_preds)}")