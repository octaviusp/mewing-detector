from PIL import Image
from os import mkdir, path, listdir
import numpy as np

IMAGE_PATH_FROM_HERE = "images/"

class ImageProcessor():
    """
        Image Processor Class.
        This class must provide an image pre-processing for further use in computer vision.
    """

    def __init__(self, dimension: tuple = (128,128), channels: int = 1):
        self.IMAGE_DIMENSION = dimension
        self.IMAGE_COUNTER = 0
        self.IMAGE_FOLDER_PATH = "./images/processed"
        self.CHANNELS = channels
        self.create_save_folder()

    def create_save_folder(self):
        """
            Creates the save folder path if not exists.
        """
        if not path.exists(self.IMAGE_FOLDER_PATH):
            mkdir(self.IMAGE_FOLDER_PATH)

    def img2np(self, image_path: str):
        """
            Turn an Image into numpy array.
        """
        img = Image.open(image_path)
        img_array = np.array(img)
        return img_array

    def process(self, image: str | bytes):
        """
            Process a image to resize and change color channels.
            The new dimension and new color channel is configured in the image processor instance.
            Finally will be saved in to the save folder path.
        """
        img = Image.open(IMAGE_PATH_FROM_HERE+str(image))
        resized_img = img.resize(self.IMAGE_DIMENSION)

        color_mode = "L" if self.CHANNELS == 1 else "RGB"
        converted_img = resized_img.convert(color_mode)

        save_path = self.IMAGE_FOLDER_PATH+"image_"+str(self.IMAGE_COUNTER)+".png"
        converted_img.save(save_path)

        return save_path

Processor_1 = ImageProcessor(dimension=(128, 128), channels=1)
