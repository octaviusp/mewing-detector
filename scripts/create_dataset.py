from PIL import Image
import os
from . import image_to_numpy_array as img2np
import random

POSITIVES_IMAGE_PATH = "./images/positives/"
NEGATIVES_IMAGE_PATH = "./images/negatives/"

def extract_label(file_name):
    name_parts = file_name.split("_")[-1]
    label = name_parts.split(".")[0]
    return int(label)

def grab_images_and_labelling(initial_path, label):
    image_names = os.listdir(initial_path)
    x = []
    y = []
    for image in image_names:
        complete_path = initial_path+image
        img_label = extract_label(image)
        nparray = img2np.img_to_np(complete_path)
        nparray = nparray.reshape(nparray.shape[0], nparray.shape[1], 1)
        x.append(nparray)
        y.append(label)

    return {
        "x": x,
        "y": y
    }


def shuffle_x_and_y(x, y):
    combined = list(zip(x, y))
    random.shuffle(combined)
    x_shuffled, y_shuffled = zip(*combined)
    return list(x_shuffled), list(y_shuffled)

def generate():
    mewing_positive = grab_images_and_labelling(POSITIVES_IMAGE_PATH, 1)
    mewing_negative = grab_images_and_labelling(NEGATIVES_IMAGE_PATH, 0)

    concatenated_x = mewing_negative["x"] + mewing_positive["x"]
    concatenated_y = mewing_negative["y"] + mewing_positive["y"]

    x, y = shuffle_x_and_y(concatenated_x, concatenated_y)

    return x,y

generate()
