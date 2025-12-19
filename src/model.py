import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from .config import DEVICE

class GeoResNet18(nn.Module):
    #ResNet-18 backbone with a custom classifier head for N countries.

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        #Use the new 'weights' API to avoid deprecation warnings
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        #Load ResNet-18 with or without pretrained ImageNet weights
        self.backbone = models.resnet18(weights=weights)

        #Replace the final fully-connected layer with our classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        #Forward pass simply delegates to the backbone
        return self.backbone(x)


def build_old_model(num_classes: int) -> GeoResNet18:
    #Build the 'old model' starting from ImageNet-pretrained ResNet-18.
    model = GeoResNet18(num_classes=num_classes, pretrained=True)
    model.to(DEVICE)
    return model


def freeze_backbone(model: GeoResNet18):
    #Freeze all layers except the final fully-connected layer.
    #Useful when doing a quick baseline or the first stage of fine-tuning.
    for name, param in model.backbone.named_parameters():
        #Keep the final classifier trainable
        if "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_backbone(model: GeoResNet18):
    #Make all layers trainable (for deeper fine-tuning).
    for param in model.parameters():
        param.requires_grad = True


def save_model(model: nn.Module, path: str):
    #Save model weights to disk.
    torch.save(model.state_dict(), path)


def load_model(path: str, num_classes: int) -> GeoResNet18:
    #Load a GeoResNet18 model from disk.
    model = GeoResNet18(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    return model
