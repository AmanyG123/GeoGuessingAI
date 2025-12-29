from pathlib import Path
import csv

from datasets import load_dataset
from PIL import Image


HF_DATASET = "deboradum/GeoGuessr-countries"
MAX_PER_COUNTRY = 20
NUM_COUNTRIES = 4
IMAGE_SIZE = (224, 224)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_CSV = DATA_DIR / "labels.csv"


def clear_old_data():
    if IMAGES_DIR.exists():
        for p in IMAGES_DIR.glob("*"):
            if p.is_file():
                p.unlink()
    if LABELS_CSV.exists():
        LABELS_CSV.unlink()


def main():
    #Demo download settings
    print("===Download subset===")
    print("Max per country:", MAX_PER_COUNTRY)
    print("Num countries:", NUM_COUNTRIES)

    #Make folders
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    #Optional:clear old data
    clear_old_data()

    #Load dataset (streaming)
    print("Loading dataset:", HF_DATASET)
    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    country_list = []
    counts = {}
    rows = []

    for example in ds:
        #Read label info
        country_name = example.get("country", None)
        if country_name is None:
            country_name = example.get("label_text", None)
        if country_name is None:
            continue

        #Pick first N countries
        if country_name not in country_list:
            if len(country_list) >= NUM_COUNTRIES:
                continue
            country_list.append(country_name)

        #Limit per country
        current_count = counts.get(country_name, 0)
        if current_count >= MAX_PER_COUNTRY:
            continue

        #Load image
        img = example["image"]
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE)

        #Save image
        filename = f"{country_name}_{current_count:04d}.jpg"
        save_path = IMAGES_DIR / filename
        img.save(save_path, format="JPEG", quality=85)

        #Track label row
        counts[country_name] = current_count + 1
        rows.append({"image": filename, "country": country_name})

        print("Saved:", filename)

        #Stop when done
        if len(country_list) == NUM_COUNTRIES:
            if all(counts.get(c, 0) >= MAX_PER_COUNTRY for c in country_list):
                break

    #Write labels.csv
    print("Writing labels.csv:", LABELS_CSV)
    with LABELS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "country"])
        writer.writeheader()
        writer.writerows(rows)

    #Print summary
    print("===Done===")
    print("Countries:", country_list)
    print("Counts:", counts)
    print("Total images:", len(rows))
    print("Data dir:", DATA_DIR)


if __name__ == "__main__":
    main()
