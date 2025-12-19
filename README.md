GeoGuessingAI – Country Prediction from Street View Images
This project trains a convolutional neural network (CNN) to guess which country a Street View–style image is from, similar to how a human plays GeoGuessr. We use a pre-trained ResNet-18 (trained on ImageNet) and compare two approaches: (1) a baseline model that only trains the final classification layer, and (2) a fine-tuned model where the entire CNN is updated on GeoGuessr-style data. The goal is to see how much performance improves when we fully fine-tune the network versus just retraining the classification head.

Project Overview
Images come from the public Hugging Face dataset "deboradum/GeoGuessr-countries". A custom script downloads a subset of images to a local "data/" folder and writes a "labels.csv" file that maps each filename to a country. The training pipeline then:
- Loads the CSV and images into a custom GeoDataset.
- Splits the data into train (70%), validation (15%), and test (15%) sets.
- Trains a baseline ResNet-18 with only the final layer trainable.
- Fine-tunes the full ResNet-18 (all layers trainable) with a smaller learning rate.
- Evaluates both models and saves:
  - overall_accuracy.csv (overall accuracy per model)
  - per_class_accuracy.csv (accuracy per country per model)
- Plots bar charts comparing baseline vs. fine-tuned performance.

Repository Structure
GeoGuessingAI/
  src/
    __init__.py
    config.py        (global settings: paths, batch size, epochs, learning rates, device)
    dataset.py       (GeoDataset class: loads images and labels, applies transforms)
    model.py         (builds ResNet-18, freeze/unfreeze functions, save/load helpers)
    train.py         (training loop for one stage: baseline or fine-tuning)
    evaluate.py      (test-time evaluation, overall and per-class accuracy)
  scripts/
    download_subset.py  (downloads a subset of the HF dataset, saves images + labels.csv)
    plot_results.py     (plots overall and per-country accuracy from CSVs)
  data/              (created by download script; contains images/ and labels.csv)
  results/           (CSV and PNG outputs written by main.py and plot_results.py)
  main.py            (orchestrates full pipeline: data loading, training, fine-tuning, evaluation)
  requirements.txt   (Python dependencies, if provided)
  README.md

Installation
1. Clone the repository:
   git clone https://github.com/AmanyG123/GeoGuessingAI.git
   cd GeoGuessingAI

2. Create and activate a virtual environment (recommended):
   python -m venv .venv
   Windows PowerShell: .venv\Scripts\activate
   macOS/Linux: source .venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt
   (If there is no requirements.txt, install at least: torch, torchvision, datasets, pandas, matplotlib, Pillow.)

Downloading a Dataset Subset
To keep storage manageable, we work with a subset of the full Hugging Face dataset. Run:
   python scripts/download_subset.py
This script:
- Streams images from "deboradum/GeoGuessr-countries".
- Downscales each image.
- Saves them into "data/images/".
- Creates "data/labels.csv" with columns:
  - image  (filename, e.g., Brazil_0003.jpg)
  - country (country name, e.g., Brazil)
You can change the subset size by editing constants in "download_subset.py" (for example, MAX_PER_COUNTRY and any country whitelist).

Running the Training Pipeline
Once labels.csv and the images are ready, run:
   python main.py
main.py will:
- Load the dataset via GeoDataset.
- Split it into train, validation, and test sets.
- Build a baseline ResNet-18 model:
  - Freeze the backbone (convolutional layers).
  - Train only the final classification layer.
  - Evaluate on the test set.
- Build a fine-tuned model:
  - Start from the trained baseline.
  - Unfreeze the backbone (all convolutional layers).
  - Continue training with a smaller learning rate.
  - Evaluate on the same test set.
- Save:
  - results/overall_accuracy.csv
  - results/per_class_accuracy.csv
  - model weights in models/ (e.g., baseline_resnet18.pth, finetuned_resnet18.pth)
The console output prints training and validation loss/accuracy for each epoch and final test accuracies for both models.

Plotting the Results
To generate plots comparing the baseline and fine-tuned models, run:
   python scripts/plot_results.py
This script reads:
- results/overall_accuracy.csv → writes results/overall_accuracy.png
- results/per_class_accuracy.csv → writes results/per_class_accuracy.png
The plots show:
- A global accuracy comparison (baseline vs. fine-tuned).
- A per-country bar chart where both accuracies are displayed side by side.

Main Results (Summary)
On a smaller initial subset of the data, the baseline model reached about 39% test accuracy, while the fine-tuned model reached about 72% accuracy on the same held-out test set. On a larger subset with all 85 countries, the baseline model achieved about 19.6% accuracy and the fine-tuned model achieved about 41.7% accuracy. Even though the task became much harder with more classes and more diverse images, fully fine-tuning the network still more than doubled the baseline accuracy. Per-country analysis shows that fine-tuning improves performance for the majority of countries, especially those that were difficult for the baseline model.

Future Extensions
Possible next steps for this project include:
- Training on a larger portion of the dataset (or the full dataset) to stabilize per-country accuracy estimates.
- Comparing different architectures (such as ResNet-34, ResNet-50, or Vision Transformers).
- Computing and visualizing a full confusion matrix to study which countries the model confuses and why.
- Using dimensionality reduction methods (like t-SNE or UMAP) to visualize how countries cluster in the model’s feature space.
These extensions would help us better understand how the model perceives the visual world and how that relates to human GeoGuessr strategies.
