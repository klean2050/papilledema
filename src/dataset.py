import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from random import uniform

from src.utils import *


class PapDataset(Dataset):
    def __init__(self, root, subjects, train):
        self.root_dir = root
        self.severity = severity
        self.dataset, self.targets = [], []
        self.names, self.sites = [], []
        self.subjects = subjects
        self.train = train

        self.transform = (
            transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            if self.train
            else transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        )

        metadata = pd.read_csv("/data/avramidi/chla_fundus/metadata.csv")
        sev_file = pd.read_csv(
            "/data/avramidi/chla_fundus/consensus_grades_severity.csv"
        )
        image_paths = [
            self.root_dir + path
            for path in os.listdir(self.root_dir)
            if path not in excluded
        ]

        for image in image_paths:
            name = image.split("/")[-1].split(".")[0].split("_")
            index, visit = int(name[0]), name[1]
            if first_visit and int(visit[-1]) != 1:
                continue

            subject_row = metadata.loc[metadata["record_id"] == index]
            collect_site = subject_row["site"].values[0] if index != 52 else 6
            camera = subject_row[f"{visit}_camera"].values[0]
            # discard optos images
            if camera == 4:
                continue
            label = subject_row["diagnosis"].values[0]
            if index not in self.subjects:
                continue

            # check severity of papilledema photos
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

            self.dataset.append(image)
            self.targets.append(int(label) - 1)
            self.sites.append(int(collect_site))
            self.names.append(image.split("/")[-1])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = plt.imread(self.dataset[idx])[..., :3]
        sample = self.apply_transforms(sample)
        for s in range(len(sample)):
            sample[s] = Image.fromarray(np.uint8(sample[s]))
            sample[s] = self.transform(sample[s])

        return sample, self.targets[idx], self.names[idx], self.sites[idx]

    def get_ratio(self):
        ratio = sum(self.targets) / (len(self.targets) - sum(self.targets))
        return ratio

    def to_csv(self):
        df = pd.DataFrame()
        df["subject"] = self.names
        df["filename"] = self.dataset
        df["site"] = self.sites
        df["label"] = self.targets
        return df

    def apply_transforms(self, img):

        red, green = img.copy(), img.copy()
        p = uniform(1.2, 1.7) if self.train else 1

        i = Image.fromarray(np.uint8(img[..., 0]))
        enhancer = ImageEnhance.Contrast(i)
        red_ch = enhancer.enhance(p)
        red[..., 0] = np.array(red_ch)

        i = Image.fromarray(np.uint8(img[..., 1]))
        enhancer = ImageEnhance.Contrast(i)
        green_ch = enhancer.enhance(p)
        green[..., 1] = np.array(green_ch)

        return [img, red, green]
