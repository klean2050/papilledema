import os, pandas as pd
from src.scripts import train_fold
from src.utils import *

os.makedirs(rest_dir, exist_ok=True)
logger = rest_dir + f"logger_{len(os.listdir(rest_dir))}.txt"

subjects = []
metadata = pd.read_csv("/data/avramidi/chla_fundus/metadata.csv")
image_paths = [data_dir + path for path in os.listdir(data_dir)]

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
    subjects.append((index, int(label) - 1, collect_site))

for fold in range(k_folds):
    print("\n=============================")
    print(f"Fold {fold}")
    print("=============================")
    train_fold(subjects, fold, logger)
