from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def plot_overall_accuracy():
    #Load overall csv
    csv_path = RESULTS_DIR / "overall_accuracy.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Run main.py first.")

    df = pd.read_csv(csv_path)

    #Plot overall accuracy
    plt.figure(figsize=(5, 4))
    plt.bar(df["model"], df["accuracy"])
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Overall Accuracy: Baseline vs Fine-tuned")

    out_path = RESULTS_DIR / "overall_accuracy.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved:", out_path)


def plot_per_class_accuracy():
    #Load per class csv
    csv_path = RESULTS_DIR / "per_class_accuracy.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Run main.py first.")

    df = pd.read_csv(csv_path)

    #Sort by fine-tuned
    df = df.sort_values("finetuned_acc", ascending=False)

    x = range(len(df))
    width = 0.4

    #Plot per country accuracy
    plt.figure(figsize=(14, 6))
    plt.bar([i - width / 2 for i in x], df["baseline_acc"], width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x], df["finetuned_acc"], width=width, label="Fine-tuned")

    plt.xticks(list(x), df["country"], rotation=90)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.title("Per-country Accuracy: Baseline vs Fine-tuned")
    plt.legend()

    out_path = RESULTS_DIR / "per_class_accuracy.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved:", out_path)


def main():
    #Make sure results folder exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading results from:", RESULTS_DIR)
    plot_overall_accuracy()
    plot_per_class_accuracy()
    print("Done.")


if __name__ == "__main__":
    main()
