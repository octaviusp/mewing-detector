from PIL import Image, ImageEnhance, ImageOps
import random
from os import mkdir, path, listdir
import numpy as np

IMAGE_PATH_FROM_HERE = "images/"

class ImageProcessor():
    """
        Image Processor Class.
        This class must provide an image pre-processing for further use in computer vision.
    """

    def __init__(self,
                dimension: tuple = (128,128),
                channels: int = 1):

        self.IMAGE_DIMENSION = dimension
        self.IMAGE_COUNTER = 0
        self.NEGATIVE_LABEL_IMAGE_FOLDER_PATH = "./images/positives/"
        self.NEGATIVE_LABEL_IMAGE_FOLDER_PATH = "./images/negatives/"
        self.CHANNELS = channels
        self.create_save_folder()

    def create_save_folder(self):
        """
            Creates the save folder path if not exists.
        """
        if not path.exists(self.NEGATIVE_LABEL_IMAGE_FOLDER_PATH):
            mkdir(self.NEGATIVE_LABEL_IMAGE_FOLDER_PATH)

    def img2np(self, image_path: str):
        """
            Turn an Image into numpy array.
        """

        img = Image.open(image_path)
        img_array = np.array(img)
        return img_array

    def process_on_demand(self, image: str | bytes):
        try:
            img = Image.open(image)
            resized_img = img.resize(self.IMAGE_DIMENSION)

            color_mode = "L" if self.CHANNELS == 1 else "RGB"
            converted_img = resized_img.convert(color_mode)
            np_array = np.array(converted_img)
            return np_array.reshape(np_array.shape[0], np_array.shape[1], self.CHANNELS)
        except Exception as e:
            pass

    def process(self, image_path: str, on_demand = False):
        try:
            img = Image.open(image_path)
            resized_img = img.resize(self.IMAGE_DIMENSION)

            color_mode = "L" if self.CHANNELS == 1 else "RGB"
            converted_img = resized_img.convert(color_mode)
            save_path = self.NEGATIVE_LABEL_IMAGE_FOLDER_PATH + "image_" + str(self.IMAGE_COUNTER) + ".png"
            converted_img.save(save_path)
            self.IMAGE_COUNTER += 1

            return save_path
        except Exception as e:
            pass

    def apply_data_augmentation(self, image_path: str):
        try:
            img = Image.open(image_path)

            # Data Augmentations
            augmentations = [
                img.transpose(Image.FLIP_LEFT_RIGHT),  # Horizontal Flip
                img.transpose(Image.FLIP_TOP_BOTTOM),  # Vertical Flip
                img.rotate(45),  # Rotate by 45 degrees
                img.rotate(90),  # Rotate by 90 degrees
                img.rotate(-30),  # Rotate by -30 degrees
                ImageEnhance.Brightness(img).enhance(1.5),  # Increase Brightness
            ]

            # Save augmented images
            augmented_images = []
            for i, augmentation in enumerate(augmentations):
                augmented_img_path = (
                    self.NEGATIVE_LABEL_IMAGE_FOLDER_PATH + f"image_{self.IMAGE_COUNTER}_{i + 1}.png"
                )
                augmentation.save(augmented_img_path)
                augmented_images.append(augmented_img_path)
                self.IMAGE_COUNTER += 1

            return augmented_images
        except:
            pass

Processor_1 = ImageProcessor(dimension=(128, 128), channels=1)
