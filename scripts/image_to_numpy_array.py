from PIL import Image
import numpy as np

# Convert the image to a NumPy array
def img_to_np(img_path: str):
    img = Image.open(img_path)
    img_array = np.array(img)
    return img_array
