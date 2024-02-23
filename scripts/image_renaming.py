import os

IMAGES_PATH = "../images/"

index = 0

print(os.listdir(IMAGES_PATH))

for image in os.listdir(IMAGES_PATH):
    complete_path = IMAGES_PATH+image
    complete_name = IMAGES_PATH+"mewing_"+str(index)+"_1.jpeg"
    print(complete_name)
    if  os.path.isdir(complete_path):
        continue
    os.rename(complete_path, complete_name)
    index += 1
