from image_processors import *
from image_operations import *
import util

class IrisSegmentator:
    def __init__(self, image):
        
        self.init_img = image
        self.extractor = IrisExtractor(self.init_img)

    def display_extraction(self):
        self.extractor.display_extraction()

class IrisExtractor:

    def __init__(self, image):
        self.init_img = image
        self.pupil_coef = 4
        self.iris_coef = 0.6
        self.extract_iris()

    def extract_iris(self):

        self.iris_boundary = self.find_iris_boundary(self.init_img)
        self.pupil_boundary = self.find_pupil_boundary(self.init_img)

        self.mask = intersect(self.iris_boundary, self.pupil_boundary)
        self.iris_extracted = apply_mask(self.init_img, self.mask)

    def find_boundary(self, img, coef):

        processor = GrayscaleProcessor()
        img = processor.process(img)

        mean_brightness = get_mean_brightness(img)
        threshold = mean_brightness / coef

        processor = BinarizationProcessor()
        processor.set_param("_threshold", threshold)

        return processor.process(img)

    def find_iris_boundary(self, img):
        img = NegativeProcessor().process(img)
        return self.find_boundary(img, self.iris_coef)

    def find_pupil_boundary(self, img):
        return self.find_boundary(img, self.pupil_coef)

    def display_extraction(self):
        self.init_img.show()
        # self.iris_boundary.show()
        # self.pupil_boundary.show()
        # self.mask.show()
        self.iris_extracted.show()


if __name__ == "__main__":
    IRIS_DIR = "teczowka_data"
    image_files = util.get_image_files(IRIS_DIR)
    image = Image.open(image_files[0])

    segmentator = IrisSegmentator(image)
    segmentator.display_extraction()
    # iris_img = segmentator.extract_iris()
