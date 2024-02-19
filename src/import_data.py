import os
from pathlib import Path
import opendatasets


def import_data():
    
    opendatasets.download("https://www.kaggle.com/datasets/tongpython/cat-and-dog")
    os.rename("cat-and-dog" , "cat_and_dog_temp")
    image_path = Path("cat_and_dog")
    image_path_list = list(image_path.glob("*/*/*/*.jpg"))
    train_dir = Path("cat_and_dog/training_set/training_set")
    test_dir = Path("cat_and_dog/test_set/test_set")
    return train_dir , test_dir
