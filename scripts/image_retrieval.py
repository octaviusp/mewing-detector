# CREATE A DATASET TO RETRIEVE MEWING PICTURES ??
from .image_processing import Processor_1
import os

def get_all_images(path: str):
    return os.listdir(path)

def process_entire_images(image_path: str):
    image_processed =  Processor_1.process(image_path)
    augmented_image_paths = None
    if image_processed is not None:
        augmented_image_paths = Processor_1.apply_data_augmentation(image_processed)
        return augmented_image_paths
    return None
