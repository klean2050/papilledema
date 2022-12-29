import os, pandas as pd
from src.scripts import train_fold

num_classes = 2
batch_size = 32
num_epochs = 50
k_folds = 10
data_dir = "data/cropped/"
feature_extract = False
optos = False
mode = "multiple"

subjects = []
metadata = pd.read_csv("/data/avramidi/chla_fundus/metadata.csv")
image_paths = [data_dir + path for path in os.listdir(data_dir)]

for image in image_paths:
    name = image.split("/")[-1].split(".")[0].split("_")
    index, visit = int(name[0]), name[1]

    subject_row = metadata.loc[metadata["record_id"] == index]
    camera = subject_row[f"{visit}_camera"].values[0]
    collect_site = subject_row["site"].values[0]

    if optos and camera != 4:
        continue
    if not optos and camera == 4:
        continue

    # papilledema: 0 | pseudo: 1
    label = subject_row["diagnosis"].values[0]
    subjects.append((index, int(label) - 1, collect_site))

for fold in range(k_folds):
    print("\n------------------------------")
    print(f"Fold {fold}")
    print("------------------------------")
    train_fold(data_dir, subjects, batch_size, num_epochs, mode, fold)
