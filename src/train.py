from typing import Dict, List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .config import DEVICE
def train_one_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    stage_name: str = "stage",
) -> Dict[str, List[float]]:
    #Generic training function for one stage (baseline or fine-tuning).
    #Returns a history dict with train/val losses and accuracies.
    criterion = nn.CrossEntropyLoss()
    #Only optimize parameters that require gradients
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_optimize, lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_train_loss = running_loss / total if total > 0 else 0.0
        epoch_train_acc = correct / total if total > 0 else 0.0

        #Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total if val_total > 0 else 0.0
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0.0

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"[{stage_name}] Epoch {epoch}/{num_epochs} "
            f"- Train loss: {epoch_train_loss:.4f}, acc: {epoch_train_acc:.4f} "
            f"- Val loss: {epoch_val_loss:.4f}, acc: {epoch_val_acc:.4f}"
        )

    return history
