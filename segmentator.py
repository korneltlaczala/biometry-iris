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
        self.pupil_coef = 0.22
        self.iris_coef = 1.25
        self.extract_iris()
    


    def extract_iris(self):
        self.grayscale_img = cv2.cvtColor(self.init_img, cv2.COLOR_BGR2GRAY)
        self.pupil_boundary = binarize(self.grayscale_img, threshold=self.pupil_coef)
        self.pupil_cleaned = clean_pupil(self.grayscale_img)
        self.pupil_circle = plot_pupil_radius(self.grayscale_img, self.pupil_cleaned)
    
        self.x, self.y, self.r = find_pupil_radius(self.pupil_cleaned)
        self.iris = binarize_iris(self.x, self.y, self.r, self.grayscale_img)
        self.iris_with_pupil = clean_iris(self.iris)
        self.iris_segmented = extract_iris(self.iris_with_pupil, self.pupil_cleaned)
        self.iris_circle, self.iris_r = plot_iris_radius(self.grayscale_img, self.iris_with_pupil, self.x, self.y)
        self.unwrapped_iris = unwrap_iris(self.grayscale_img, self.x, self.y, self.r, self.iris_r)
        self.iris_with_rings = draw_iris_rings(self.grayscale_img, self.x, self.y, self.r, self.iris_r)




        

    
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

      

        images = [
            self.init_img,
            self.pupil_boundary,
            self.pupil_cleaned,
            self.pupil_circle,
            self.iris,
            self.iris_with_pupil,
            self.iris_segmented,
            self.iris_circle,
            self.iris_with_rings,
            self.unwrapped_iris
           
            
        ]
        descriptions = [
            "Original Image",
            "Pupil Boundary",
            "Pupil Cleaned",
            "Pupil Circle",
            "Iris ",
            "Iris with Pupil",
            "Iris Cleaned",
            "Iris Circle",
            "Iris with Rings",
            "Unwrapped Iris"
           
        ]

        # self.plot_images(images, descriptions, rows=2)
        self.plot_images(images, descriptions, rows=2)


    def plot_images(self, images, descriptions, rows=2):
        cols = len(images) // rows
        fig, ax = plt.subplots(rows, cols, figsize=(20, 10))
        for i in range(len(images)):
            if rows == 1:
                ax[i].imshow(images[i], cmap='gray')
                ax[i].set_title(descriptions[i])
                continue
            row = i // cols
            col = i % cols
            ax[row, col].imshow(images[i], cmap='gray')
            ax[row, col].set_title(descriptions[i])
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    IRIS_DIR = "teczowka_data"
    image_files = util.get_image_files(IRIS_DIR)
    image = Image.open(image_files[0])

    segmentator = IrisSegmentator(image)
    segmentator.display_extraction()
    # iris_img = segmentator.extract_iris()
