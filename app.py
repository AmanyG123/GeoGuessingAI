from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

#Paths
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"


def load_overall_accuracy():
    path = RESULTS_DIR / "overall_accuracy.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_per_class_accuracy():
    path = RESULTS_DIR / "per_class_accuracy.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    st.title("GeoGuessr Country Classifier")
    st.write(
        "This app shows the performance of our convolutional neural network "
        "that predicts the country from Street View–style images."
    )

    overall_df = load_overall_accuracy()
    per_class_df = load_per_class_accuracy()

    if overall_df is None:
        st.error("overall_accuracy.csv not found in results/. Run main.py first.")
        return

    st.header("Overall Accuracy: Baseline vs Fine-tuned")

    try:
        baseline_acc = float(
            overall_df.loc[overall_df["model"] == "baseline", "accuracy"].iloc[0]
        )
        finetuned_acc = float(
            overall_df.loc[overall_df["model"] == "finetuned", "accuracy"].iloc[0]
        )
    except Exception:
        st.error("Could not read baseline/finetuned rows from overall_accuracy.csv.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Baseline Accuracy", f"{baseline_acc:.3f}")
    with col2:
        st.metric("Fine-tuned Accuracy", f"{finetuned_acc:.3f}")

    png_path = RESULTS_DIR / "overall_accuracy.png"
    if png_path.exists():
        st.image(str(png_path), caption="Overall accuracy bar plot", use_container_width=True)

    st.header("Per-country Accuracy")

    if per_class_df is not None:
        per_class_df = per_class_df.sort_values("finetuned_acc", ascending=False)
        st.dataframe(per_class_df, use_container_width=True)

        png_per_class = RESULTS_DIR / "per_class_accuracy.png"
        if png_per_class.exists():
            st.image(
                str(png_per_class),
                caption="Per-country accuracy: baseline vs fine-tuned",
                use_container_width=True,
            )
    else:
        st.info("per_class_accuracy.csv not found. Run main.py and plot_results.py to generate it.")

    st.markdown("---")
    st.subheader("Next Steps")
    st.write(
        "- Add an interactive view that shows example images from each country.\n"
        "- Compare individual predictions from the baseline and fine-tuned models.\n"
        "- Highlight images where the fine-tuned model fixes baseline mistakes."
    )


if __name__ == "__main__":
    main()
