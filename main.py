import os, pandas as pd
from src import *

# Set up the logger
os.makedirs(rest_dir, exist_ok=True)
logger = rest_dir + f"logger_{len(os.listdir(rest_dir))}.txt"

# Load metadata
metadata = pd.read_csv("/data/avramidi/chla_fundus/metadata.csv")
sev_file = pd.read_csv("/data/avramidi/chla_fundus/consensus_grades_severity.csv")
image_paths = [data_dir + path for path in os.listdir(data_dir) if path not in excluded]

# Determine all subjects
subjects = []
for image in image_paths:
    name = image.split("/")[-1].split(".")[0].split("_")
    index, visit = int(name[0]), name[1]

    subject_row = metadata.loc[metadata["record_id"] == index]
    camera = subject_row[f"{visit}_camera"].values[0]
    collect_site = subject_row["site"].values[0] if index != 52 else 6

    if optos and camera != 4:
        continue
    if not optos and camera == 4:
        continue

    # papilledema: 0 | pseudo: 1
    label = subject_row["diagnosis"].values[0]
    if int(label) == 1:
        try:
            this_severity = sev_file.loc[
                sev_file["original name"] == image.split("/")[-1]
            ]["consensus_grade"].values[0]
        except:
            continue
        if severity == "mild" and this_severity > 2:
            continue
        if severity == "severe" and this_severity < 3:
            continue
    subjects.append((index, int(label) - 1, collect_site))

# Iterate over training folds
subjects, raw = list(set(subjects)), {}
for fold in range(k_folds):
    print("\n=============================")
    print(f"Fold {fold}")
    print("=============================")
    raw = train_fold(subjects, fold, logger, raw)

# Curate and save the results
csv_data = pd.DataFrame()
csv_data["image"] = list(raw.keys())
labels = [v[0][1] for _, v in raw.items()]
csv_data["PA_proba"] = [v[0][0] for _, v in raw.items()]
csv_data["PA_label"] = [False if l else True for l in labels]
csv_data.to_csv(f"predictions/saved_dict_{test_site}.csv")
