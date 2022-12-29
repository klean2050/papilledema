import os, numpy as np, cv2 as cv
import matplotlib.pyplot as plt, pandas as pd
from skimage import measure
from skimage.exposure import equalize_adapthist
from tqdm import tqdm

path = "/data/avramidi/chla_fundus/"
metadata = pd.read_csv(path + "metadata.csv")
image_paths = [p for p in os.listdir(path) if not p.endswith("csv")]

excluded = [
    # show atrophy, have resolved PA
    "198_visit01_photo01.JPG",
    "211_visit01_photo01.JPG",
    "224_visit01_photo01.jpg",
    "224_visit01_photo02.jpg",
    "224_visit02_photo01.jpg",
    "224_visit03_photo01.jpg",
    "224_visit04_photo01.jpg",
    "224_visit05_photo01.jpg",
    "224_visit07_photo01.jpg",
    "224_visit10_photo01.jpg",
    "235_visit01_photo01.jpg",
    "235_visit01_photo02.jpg",
    "235_visit02_photo01.jpg",
    "235_visit02_photo02.jpg",
    "235_visit03_photo01.jpg",
    "235_visit03_photo02.jpg",
    "307_visit01_photo02.JPG",
]

optos = []
for image in image_paths:
    name = image.split("/")[-1].split(".")[0].split("_")
    index, visit = int(name[0]), name[1]
    subject_row = metadata.loc[metadata["record_id"] == index]
    camera = subject_row[f"{visit}_camera"].values[0]
    if camera == 4:
        optos.append(image)

image_paths = [i for i in image_paths if i not in excluded and i not in optos]

stgs = {"op": 5, "cl": 10}
kernel = lambda x: np.ones((x, x), np.uint8)

out_folder = "data/cropped/"
os.makedirs(out_folder, exist_ok=True)
for i in tqdm(os.listdir(path)):

    if i not in image_paths or i in os.listdir(out_folder):
        continue
    j = cv.imread(path + i)
    im = j[..., 2]  # R

    t = 0.2 * np.mean(im.ravel())
    _, img = cv.threshold(im, int(t), 255, cv.THRESH_BINARY)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel(stgs["op"]))
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel(stgs["cl"]))

    regions = measure.regionprops(img)
    rbubble = regions[0]
    y0, x0 = rbubble.centroid
    r = rbubble.major_axis_length / 2.0

    Y, X = np.ogrid[: im.shape[0], : im.shape[1]]
    dist_from_center = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    mask = dist_from_center <= 0.8 * r
    im[~mask] = 0

    ig = im.copy()
    ig = cv.blur(ig, (31, 31), 0)

    list_intens = sorted(ig.ravel())
    t = list_intens[int(0.99 * len(list_intens))]
    _, img = cv.threshold(ig, int(t), 255, cv.THRESH_BINARY)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel(stgs["op"]))
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel(stgs["cl"] * 5))

    labeled_img, marea = measure.label(img), 2000
    for l in measure.regionprops(labeled_img):
        if l.eccentricity < 0.9 and l.area > marea:
            selected = l
            marea = l.area

    m = plt.imread(path + i)  # RGB
    m = equalize_adapthist(m)
    try:
        y0, x0, y, x = selected.bbox
        air = int(max(y - y0, x - x0) * 0.8)
        plt.imsave(out_folder + i, m[y0 - air : y + air, x0 - air : x + air])
    except:
        # save cropped image from step 1
        y0, x0, y, x = rbubble.bbox
        air = -int(max(y - y0, x - x0) * 0.2)
        plt.imsave(out_folder + i, m[y0 - air : y + air, x0 - air : x + air])
