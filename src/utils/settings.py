# system parameters
data_dir = "data/cropped/"
rest_dir = "logging/"
devices = "1"

# split parameters
optos = False
n_turns = 5
k_folds = 1
test_size = 0.2
per_site = True
external = False
test_site = 1.0  # should be in [1.0, 2.0, 5.0, 6.0]
first_visit = False
severity = "none"

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
    # new excluded photos
    "41_visit03_photo01.jpg",
    "58_visit03_photo01.jpg",
    "224_visit05_photo02.jpg",
    "247_visit01_photo01.jpg",
    "277_visit03_photo01.JPG",
]
