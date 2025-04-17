from image_processors import *

class IrisSegmentator:
    def __init__(self, image):
        
        self.init_img = image

    def extract_iris(self):
        flow = ProcessorFlow()
        grayscale = GrayscaleProcessor()
        binarization = BinarizationProcessor()
        grayscale.set_param("_is_enabled", True)
        binarization.set_param("_is_enabled", True)
        flow.add_processor(grayscale)
        flow.add_processor(binarization)
        img = flow.process(self.init_img)
        return img
        