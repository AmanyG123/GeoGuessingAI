import os
import csv

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.config import (
    LABELS_CSV,
    BATCH_SIZE,
    NUM_EPOCHS_BASELINE,
    NUM_EPOCHS_FINETUNE,
    LEARNING_RATE_BASELINE,
    LEARNING_RATE_FINETUNE,
    DEVICE,
)
from src.dataset import GeoDataset
from src.model import (
    build_old_model,
    freeze_backbone,
    unfreeze_backbone,
    save_model,
    load_model,
)
from src.train import train_one_stage
from src.evaluate import evaluate_model

def create_dataloaders(csv_path: str, batch_size: int):

 #   Create train/val/test DataLoaders from the CSV.

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #Normalization values for ImageNet-pretrained ResNet
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = GeoDataset(csv_path=csv_path, transform=transform)

    num_samples = len(dataset)
    if num_samples < 3:
        raise ValueError("Dataset is too small. Add more images/rows to labels.csv.")

    #70% train, 15% val, 15% test
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return dataset, train_loader, val_loader, test_loader


def main():
    #Create dataloaders
    dataset, train_loader, val_loader, test_loader = create_dataloaders(
        LABELS_CSV, BATCH_SIZE
    )
    num_classes = len(dataset.countries)
    class_names = dataset.countries

    print(f"Total images: {len(dataset)}")
    print(f"Number of countries: {num_classes}")
    print("Countries:", class_names)

    #Directory to save models
    project_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    baseline_path = os.path.join(models_dir, "baseline_resnet18.pth")
    finetuned_path = os.path.join(models_dir, "finetuned_resnet18.pth")

    #1: Builds and trains the whole model
    print("\n=== Building baseline (old) model ===")
    model = build_old_model(num_classes=num_classes)

    #Freeze backbone, train only the final layer for a short stage
    freeze_backbone(model)
    print("Training baseline (only final layer)...")
    train_one_stage(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS_BASELINE,
        learning_rate=LEARNING_RATE_BASELINE,
        stage_name="baseline",
    )

    print("Evaluating baseline model on test set...")
    baseline_results = evaluate_model(model, test_loader, class_names)
    print(f"Baseline accuracy: {baseline_results['accuracy']:.4f}")
    save_model(model, baseline_path)
    print(f"Saved baseline model to {baseline_path}")

    #Fine-tunes the new model
    print("\n=== Fine-tuning model (new model) ===")

    #Loads the baseline model (just to be explicit)
    model_finetune = load_model(baseline_path, num_classes=num_classes)


    #Unfreeze entire backbone for deeper fine-tuning
    unfreeze_backbone(model_finetune)

    train_one_stage(
        model_finetune,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS_FINETUNE,
        learning_rate=LEARNING_RATE_FINETUNE,
        stage_name="finetune",
    )

    print("Evaluating fine-tuned model on test set...")
    finetune_results = evaluate_model(model_finetune, test_loader, class_names)
    print(f"Fine-tuned accuracy: {finetune_results['accuracy']:.4f}")
    save_model(model_finetune, finetuned_path)
    print(f"Saved fine-tuned model to {finetuned_path}")

    #3: Compares the results for Baseline and Finetune
    print("\n=== Comparison: Baseline vs Fine-tuned ===")
    print(f"Baseline accuracy:   {baseline_results['accuracy']:.4f}")
    print(f"Fine-tuned accuracy: {finetune_results['accuracy']:.4f}")

    print("\nPer-class accuracy (baseline):")
    for cls, acc in baseline_results["per_class_accuracy"].items():
        print(f"  {cls:20s}: {acc:.4f}")

    print("\nPer-class accuracy (fine-tuned):")
    for cls, acc in finetune_results["per_class_accuracy"].items():
        print(f"  {cls:20s}: {acc:.4f}")

    #4: Saves the Result to CSV for plotting
    results_dir = os.path.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    #Saves overall accuracies
    overall_path = os.path.join(results_dir, "overall_accuracy.csv")
    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy"])
        writer.writerow(["baseline", baseline_results["accuracy"]])
        writer.writerow(["finetuned", finetune_results["accuracy"]])

    #Saves per-class accuracies
    per_class_path = os.path.join(results_dir, "per_class_accuracy.csv")
    with open(per_class_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["country", "baseline_acc", "finetuned_acc"])

        baseline_per_class = baseline_results["per_class_accuracy"]
        finetune_per_class = finetune_results["per_class_accuracy"]
        all_classes = sorted(set(baseline_per_class.keys()) | set(finetune_per_class.keys()))

        for cls in all_classes:
            writer.writerow([
                cls,
                baseline_per_class.get(cls, 0.0),
                finetune_per_class.get(cls, 0.0),
            ])

    print(f"\nSaved CSV results to: {results_dir}")
    print("\nDone.")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    main()
