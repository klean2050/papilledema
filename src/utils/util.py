# path parameters
data_dir = "data/cropped/"
rest_dir = "results/"

# split parameters
optos = False
n_turns = 5
k_folds = 10
test_size = 0.2
per_site = False
test_site = 6.0
first_visit = False

# train parameters
num_epochs = 50
batch_size = 32
learning_r = 1e-4
ddloss = "sites"
mode = "multiple"

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
