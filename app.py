import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st

#Setprojectpaths
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

def load_results():
    #LoadCSVresults
    overall_path = RESULTS_DIR / "overall_accuracy.csv"
    per_class_path = RESULTS_DIR / "per_class_accuracy.csv"

    if not overall_path.exists() or not per_class_path.exists():
        st.error("Results CSV files not found. Please run main.py first to generate results.")
        return None, None

    overall_df = pd.read_csv(overall_path)
    per_class_df = pd.read_csv(per_class_path)

    #Addimprovementcolumn
    per_class_df["improvement"] = per_class_df["finetuned_acc"] - per_class_df["baseline_acc"]

    return overall_df, per_class_df

def show_overall(overall_df):
    #Showoverallaccuracy
    st.subheader("Overall Accuracy")

    baseline_row = overall_df[overall_df["model"] == "baseline"]
    finetuned_row = overall_df[overall_df["model"] == "finetuned"]

    baseline_acc = float(baseline_row["accuracy"].values[0])
    finetuned_acc = float(finetuned_row["accuracy"].values[0])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Baseline accuracy", f"{baseline_acc:.3f}")
    with col2:
        st.metric("Fine-tuned accuracy", f"{finetuned_acc:.3f}")
    with col3:
        st.metric("Improvement", f"{(finetuned_acc - baseline_acc):.3f}")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(overall_df["model"], overall_df["accuracy"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Baseline vs Fine-tuned (Overall)")
    st.pyplot(fig)

def show_per_country(per_class_df):
    #Showpercountryresults
    st.subheader("Per-country Accuracy")

    sort_option = st.selectbox(
        "Sort countries by",
        ["Fine-tuned accuracy (high→low)", "Improvement (high→low)", "Baseline accuracy (high→low)", "Country name (A→Z)"],
    )

    if sort_option.startswith("Fine-tuned"):
        df = per_class_df.sort_values("finetuned_acc", ascending=False)
    elif sort_option.startswith("Improvement"):
        df = per_class_df.sort_values("improvement", ascending=False)
    elif sort_option.startswith("Baseline"):
        df = per_class_df.sort_values("baseline_acc", ascending=False)
    else:
        df = per_class_df.sort_values("country", ascending=True)

    max_countries = len(df)
    if max_countries == 0:
        st.warning("No per-country results found.")
        return

    min_countries = 1
    default_n = min(20, max_countries)

    top_n = st.slider(
        "How many countries to show",
        min_value=min_countries,
        max_value=max_countries,
        value=default_n,
    )

    df_top = df.head(top_n)

    st.write("Table of per-country accuracy:")
    st.dataframe(df_top[["country", "baseline_acc", "finetuned_acc", "improvement"]])

    x = range(len(df_top))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([i - width / 2 for i in x], df_top["baseline_acc"], width=width, label="Baseline")
    ax.bar([i + width / 2 for i in x], df_top["finetuned_acc"], width=width, label="Fine-tuned")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df_top["country"], rotation=90)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-country Accuracy: Baseline vs Fine-tuned")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)


def show_country_details(per_class_df):
    #Showonecountrysummary
    st.subheader("Country Details")

    country_list = per_class_df["country"].tolist()
    selected = st.selectbox("Select a country", country_list)

    row = per_class_df[per_class_df["country"] == selected].iloc[0]
    baseline = float(row["baseline_acc"])
    finetuned = float(row["finetuned_acc"])
    improvement = float(row["improvement"])

    st.write(f"**Country:** {selected}")
    st.write(f"- Baseline accuracy: `{baseline:.3f}`")
    st.write(f"- Fine-tuned accuracy: `{finetuned:.3f}`")
    st.write(f"- Improvement: `{improvement:.3f}`")

    if improvement > 0.05:
        st.info("Fine-tuning helped a lot for this country, meaning the model learned better country-specific visual cues.")
    elif improvement < -0.05:
        st.warning("Accuracy dropped after fine-tuning. This can happen when the decision boundary shifts or there are few test images.")
    else:
        st.write("Fine-tuning did not change performance much for this country.")

def main():
    #MainStreamlitapp
    st.title("GeoGuessr Country CNN – Results Viewer")
    st.write("This app shows how our baseline and fine-tuned ResNet-18 models perform on GeoGuessr-style country prediction.")

    overall_df, per_class_df = load_results()
    if overall_df is None or per_class_df is None:
        return

    show_overall(overall_df)
    st.markdown("---")
    show_per_country(per_class_df)
    st.markdown("---")
    show_country_details(per_class_df)

if __name__ == "__main__":
    main()
