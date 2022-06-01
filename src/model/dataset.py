import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from random import uniform


class PapDataset(Dataset):
    def __init__(self, root, subjects, train):
        self.root_dir = root
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.dataset = []
        self.targets = []
        self.subjects = subjects
        self.names = []
        self.train = train

        metadata = pd.read_csv("/".join(self.root_dir.split("/")[:-2]) + "/metadata.csv")
        image_paths = [self.root_dir + path for path in os.listdir(self.root_dir)]

        for image in image_paths:
            name = image.split("/")[-1].split(".")[0].split("_")
            subject_row = metadata.loc[metadata["record_id"] == int(name[0])]
            camera = subject_row["visit01_camera"].values[0]
            if camera == 4:
                # discard optos images
                continue
            label = subject_row["diagnosis"].values[0]
            label = int(label) - 1
            if int(name[0]) not in self.subjects:
                continue

            self.dataset.append(plt.imread(image))  # RGB
            self.targets.append(label)
            self.names.append("_".join(name))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]
        sample = self.apply_transforms(sample)
        for s in range(len(sample)):
            sample[s] = Image.fromarray(np.uint8(sample[s]))
            sample[s] = self.transform(sample[s])

        return sample, self.targets[idx], self.names[idx]

    def apply_transforms(self, img):

        red, green = img.copy(), img.copy()
        p = uniform(1.6, 1.8) if self.train else 1.6

        i = Image.fromarray(np.uint8(img[..., -3]))
        enhancer = ImageEnhance.Contrast(i)
        red_ch = enhancer.enhance(p)
        red[..., -3] = np.array(red_ch)

        i = Image.fromarray(np.uint8(img[..., -2]))
        enhancer = ImageEnhance.Contrast(i)
        green_ch = enhancer.enhance(p)
        green[..., -2] = np.array(green_ch)

        return [img, red, green]
