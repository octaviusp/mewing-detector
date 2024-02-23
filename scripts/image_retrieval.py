# CREATE A DATASET TO RETRIEVE MEWING PICTURES ??
from .image_processing import Processor_1
import os

def get_all_images(path: str):
    return os.listdir(path)

def turn_into_numpy_array(image_path: str):
    image_processed =  Processor_1.process(image_path)
    img2np = Processor_1.img2np(image_processed)
    return img2np
