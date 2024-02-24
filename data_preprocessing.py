import os
from scripts import (
    image_retrieval,
    image_renaming,
    image_to_numpy_array
)

IMAGES_PATH_FROM_START_SCRIPT = "images/"
IMAGES_PATH_FOR_SCRIPTS = "/images/"
IMAGE_DIMENSION = (128,128,1)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

STEPS = [
    "- STEP 1 - RENAMING ALL IMAGES IN THE FOLDER...",
    "- STEP 2 - DOING DATA AUGMENTATION...",
    "- STEP 3 - TURNING IMAGES INTO NUMPY-ARRAY..."
]

def is_image(file_name):
    return file_name.split(".")[-1] in ALLOWED_EXTENSIONS

def extract_label(file_name):
    name_parts = file_name.split("_")[-1]
    label = name_parts.split(".")[0]
    return label

def x_y_images():
    try:
        image_renaming.rename()
    except Exception as e:
        print(e)
        print("- An error was ocurred in renaming images, please try again.")

    file_images = image_retrieval.get_all_images(IMAGES_PATH_FROM_START_SCRIPT)
    file_images = list(filter(is_image, file_images))

    images_as_numpy_arrays = []
    labels = []

    script_directory = os.path.dirname(os.path.abspath(__file__))

    for image in file_images:
        abs_path = script_directory + IMAGES_PATH_FOR_SCRIPTS + image
        label = int(extract_label(abs_path))
        image = image_retrieval.process_entire_images(abs_path)
        if image is not None:
            for img in image:
                np_array_for_image = image_to_numpy_array.img_to_np(img)
                np_array = np_array_for_image.reshape(IMAGE_DIMENSION)
                images_as_numpy_arrays.append(np_array)
                labels.append(label)

    return { "x": images_as_numpy_arrays, "y": labels }

if __name__ == "__main__":
    data = x_y_images()
