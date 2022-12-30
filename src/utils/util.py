import os, torch

batch_size = 32
num_epochs = 50
k_folds = 10
optos = False
mode = "multiple"
ddloss = "sites"

data_dir = "data/cropped/"
rest_dir = "results/"
test_size = 0.15

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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