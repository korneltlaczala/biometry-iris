from segmentator import *
import util
import os


IRIS_DIR = "teczowka_data"
image_files = util.get_image_files(IRIS_DIR)
image = Image.open(image_files[0])

segmentator = IrisSegmentator(image)
iris_img = segmentator.extract_iris()
iris_img.show()
