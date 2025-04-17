import os

def get_image_files(dir_name):
    image_files = []
    for dir in os.listdir(dir_name):
        root = os.path.join(dir_name, dir)
        for file in os.listdir(os.path.join(dir_name, dir)):
            file = file.lower()
            if file.endswith(".jpg") or file.endswith(".png"):
                image_files.append(os.path.join(root, file))
    return image_files