import os, numpy as np, cv2 as cv
import matplotlib.pyplot as plt, pandas as pd
from skimage import measure
from tqdm import tqdm
from PIL import Image, ImageEnhance

p = "../data/"
metadata = pd.read_csv(p + "metadata.csv")
image_paths = [p + "redcap_img/" + path for path in os.listdir(p + "redcap_img/")]

optos = []
for image in image_paths:
    name = image.split("/")[-1].split(".")[0].split("_")
    index, visit = int(name[0]), name[1]
    subject_row = metadata.loc[metadata["record_id"] == index]
    camera = subject_row[f"{visit}_camera"].values[0]
    if camera == 4:
        optos.append(image.split("/")[-1])

stgs = {"op": 30, "cl": 50}
kernel = lambda x: np.ones((x, x), np.uint8)
out_folder = "../data/cropped_img/"

for i in tqdm(os.listdir(p)):

    if i in optos:
        continue
    j = cv.imread(p + i)  # BGR

    reg = []
    for channel in [-1, -3]:
        im = j[..., channel]

        image = Image.fromarray(im)
        enhancer = ImageEnhance.Contrast(image)
        im = enhancer.enhance(1.5)
        im = np.array(im)

        t = 0.1 * np.mean(im.ravel())
        _, img = cv.threshold(im, int(t), 255, cv.THRESH_BINARY)
        img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel(stgs["op"] // 5))
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel(stgs["cl"] * 5))

        regions = measure.regionprops(img)
        rbubble = regions[0]
        y0, x0 = rbubble.centroid
        r = rbubble.major_axis_length / 2.0

        Y, X = np.ogrid[: im.shape[0], : im.shape[1]]
        dist_from_center = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        mask = dist_from_center <= 0.8 * r
        im[~mask] = 0

        ig = im.copy()
        ig = cv.blur(ig, (71, 71), 0)
        t = 3.5 * np.mean(ig.ravel())
        if t > 250:
            continue

        _, img = cv.threshold(ig, int(t), 255, cv.THRESH_BINARY)
        img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel(stgs["op"]))
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel(stgs["cl"]))

        labeled_img = measure.label(img)
        regions = measure.regionprops(labeled_img)
        areas = [l.area for l in regions if l.eccentricity < 0.95 and l.area > 1000]

        try:
            bubble = regions[np.argmax(areas)]
            y0, x0, y, x = bubble.bbox
            m = plt.imread(p + i)  # RGB
            plt.imsave(out_folder + i, m[y0 - 300 : y + 300, x0 - 300 : x + 300])
            break
        except:
            if channel == -3:
                # save cropped image from step 1
                y0, x0, y, x = rbubble.bbox
                m = plt.imread(p + i)  # RGB
                try:
                    plt.imsave(out_folder + i, m[y0 + 500 : y - 500, x0 + 500 : x - 500])
                except:
                    # if margins do not meet
                    plt.imsave(out_folder + i, im)
                    continue
            else:
                # continue to channel -3
                continue
