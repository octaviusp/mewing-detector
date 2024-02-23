from scripts import (
    image_retrieval
)

IMAGES_PATH_FROM_START_SCRIPT = "images/"
IMAGES_PATH_FOR_SCRIPTS = "images/"

def is_image(file_name):
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return file_name.split(".")[-1] in allowed_extensions

def extract_label(file_name):
    name_parts = file_name.split("_")[1]
    label = name_parts.split(".")[0]
    return label

def x_y_images():
    file_images = image_retrieval.get_all_images(IMAGES_PATH_FROM_START_SCRIPT)
    file_images = list(filter(is_image, file_images))

    images_as_numpy_arrays = []
    labels = []

    for image in file_images:
        label = extract_label(image)
        np_array = image_retrieval.turn_into_numpy_array(image)
        np_array = np_array.reshape((128,128,1))
        images_as_numpy_arrays.append(np_array)
        labels.append(label)

    try:
        assert len(labels) == len(images_as_numpy_arrays)
    except:
        print(" - Dimension error: Labels array must be the same length of images as numpy arrays. Check this.")

    return { "x": images_as_numpy_arrays, "y": labels }

if __name__ == "__main__":
    print(x_y_images())
