import torch, clip, os, numpy as np
import matplotlib.pyplot as plt, cv2
import pandas as pd, pickle
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve


def get_embeddings(path="data/cropped/"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    feats = []
    for i in tqdm(os.listdir(path)):
        img_path = os.path.join(path, i)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feats.append(model.encode_image(image))

    # feats.shape = (862, 512)
    return torch.vstack(feats)


def process_saliency(path="data/cropped/", smap_path="saliency/"):
    for smap in tqdm(os.listdir(smap_path)):
        smap_path = os.path.join(smap_path, smap)
        img = "_".join(smap.split("_")[1:4])
        (dim1, dim2) = plt.imread(path + img).shape[:-1]

        smap_img = plt.imread(smap_path)

        kernel = np.ones((2, 2), np.uint8)
        out = cv2.dilate(smap_img, kernel, iterations=1)
        out = cv2.GaussianBlur(out, (3, 3), 0)
        out = cv2.resize(out, (dim2, dim1), interpolation=cv2.INTER_AREA)

        smap_path = smap_path.replace("png", "jpg").replace("cy", "cy2")
        plt.imsave(smap_path, (out * 255).astype(np.uint8))


def calc_ci(distr, p):
    dmean = np.mean(distr)
    ci_do = np.percentile(distr, p / 2)
    ci_up = np.percentile(distr, 100 - p / 2)
    return "{:.1f}% ({}% CI, {:.1f} to {:.1f})".format(dmean, 100 - p, ci_do, ci_up)


def get_results(logger):
    auc, sns, acc, spc = [], [], [], []
    with open(logger, "r") as f:
        for line in f.readlines():
            if line.startswith("AUC"):
                n = line.split(":")[1][1:5]
                auc.append(float(n))
            if line.startswith("Acc"):
                n = line.split(":")[1][1:5]
                acc.append(float(n))
            if line.startswith("Sen"):
                n = line.split(":")[1][1:5]
                sns.append(float(n))
            if line.startswith("Spe"):
                n = line.split(":")[1][1:5]
                spc.append(float(n))

    print("AUC =", calc_ci(auc, p=5))
    print("Acc =", calc_ci(acc, p=5))
    print("Sns =", calc_ci(sns, p=5))
    print("Spc =", calc_ci(spc, p=5))
    return auc, acc, sns, spc


def get_predictions(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    csv_data = pd.DataFrame()
    csv_data["image"] = list(data.keys())
    labels = [v[0][1] for _, v in data.items()]

    csv_data["PA_proba"] = [v[0][0] for _, v in data.items()]
    csv_data["PA_label"] = [False if l else True for l in labels]
    csv_data.to_csv(pkl_path.replace("pkl", "csv"))


def plot_roc_curve(csv_path):
    data = pd.read_csv(csv_path)
    fpr, tpr, t = roc_curve(data["PA_label"], data["PA_proba"])
    label = roc_auc_score(data["PA_label"], data["PA_proba"])

    plt.figure(figsize=(4.5, 4.5), dpi=100)
    plt.plot(fpr, tpr, label="Ours: AUC = {:.2}".format(label), color="blue")
    plt.plot([0, 1], [0, 1], "r", label="Random Classifier")
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.01, 0.1))
    plt.grid(linestyle="--")
    plt.legend(loc="lower right")
    plt.title("Optimal Threshold: {:.2}".format(t[np.argmax(tpr - fpr)]))
    plt.show()
    plt.savefig(csv_path.replace("csv", "png"), bbox_inches="tight", dpi=100)
