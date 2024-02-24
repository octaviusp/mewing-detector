import os

IMAGES_PATH = "./images/"

def rename():
    for index, image in enumerate(os.listdir(IMAGES_PATH)):
        complete_path = IMAGES_PATH+image
        complete_name = IMAGES_PATH+"mewing_"+str(index)+"_0.jpeg"
        if  os.path.isdir(complete_path):
            continue
        os.rename(complete_path, complete_name)
