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
        self.pupil_boundary_eroded = erode(self.pupil_boundary, iterations=5)
        self.pupil_boundary_dilated = dilate(self.pupil_boundary, iterations=5)
        self.pupil_boundary_opened = open(self.pupil_boundary, iterations=5)
        self.pupil_boundary_closed = close(self.pupil_boundary, iterations=15)
        self.pupil_boundary_closed_dilated = dilate(self.pupil_boundary_closed, iterations=15)
        self.pupil_boundary_closed_eroded = erode(self.pupil_boundary_closed, iterations=15)

        self.mask = intersect(self.iris_boundary, self.pupil_boundary)
        self.iris_extracted = apply_mask(self.init_img, self.mask)
        self.iris_extracted_opened = open(self.iris_extracted, iterations=5)
        # self.mask_eroded = erode(self.mask, iterations=5)

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

        images = [self.init_img,
                #   self.iris_boundary,
                  self.pupil_boundary,
                #   self.pupil_boundary_eroded,
                #   self.pupil_boundary_dilated,
                #   self.pupil_boundary_opened,
                #   self.pupil_boundary_closed,
                #   self.pupil_boundary_closed_dilated,
                  self.pupil_boundary_closed_eroded,
                #   self.mask,
                #   self.iris_extracted,
                #   self.iris_extracted_opened,
                  ]

        descriptions = ["Initial image",
                        # "Iris boundary",
                        "Pupil boundary",
                        # "Pupil boundary eroded",
                        # "Pupil boundary dilated",
                        # "Pupil boundary opened",
                        # "Pupil boundary closed",
                        # "Pupil boundary closed dilated",
                        "Pupil boundary closed eroded",
                        # "Mask",
                        # "Iris extracted",
                        # "Iris extracted opened",
                        ]

        # self.plot_images(images, descriptions, rows=2)
        self.plot_images(images, descriptions, rows=1)


    def plot_images(self, images, descriptions, rows=1):
        cols = len(images) // rows
        fig, ax = plt.subplots(rows, cols, figsize=(20, 10))
        for i in range(len(images)):
            if rows == 1:
                ax[i].imshow(images[i])
                ax[i].set_title(descriptions[i])
                continue
            row = i // cols
            col = i % cols
            ax[row, col].imshow(images[i])
            ax[row, col].set_title(descriptions[i])
        plt.show()



if __name__ == "__main__":
    IRIS_DIR = "teczowka_data"
    image_files = util.get_image_files(IRIS_DIR)
    image = Image.open(image_files[0])

    segmentator = IrisSegmentator(image)
    segmentator.display_extraction()
    # iris_img = segmentator.extract_iris()
