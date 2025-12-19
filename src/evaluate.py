from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from .config import DEVICE


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    class_names,
) -> Dict[str, Any]:
    #Evaluate a model on a given DataLoader.
    model.eval()
    num_classes = len(class_names)

    correct = 0
    total = 0
    #Confusion matrix: rows = true, cols = predicted
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1

    accuracy = correct / total if total > 0 else 0.0

    #Per-class accuracy
    per_class_accuracy = {}
    for idx, class_name in enumerate(class_names):
        true_positives = confusion[idx, idx].item()
        total_for_class = confusion[idx, :].sum().item()
        if total_for_class > 0:
            per_class_accuracy[class_name] = true_positives / total_for_class
        else:
            per_class_accuracy[class_name] = 0.0

    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),  # convert to plain Python list
    }
