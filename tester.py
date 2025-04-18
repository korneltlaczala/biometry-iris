from segmentator import *
from PIL import Image
import util

class Tester:
    
    def test(self, DIR):
        self.image_files = util.get_files(DIR, "jpg")[:3]
        self.resuts = []
        print(f"Found {len(self.image_files)} images")
        for i, file in enumerate(self.image_files):
            print(f"Testing image {i + 1}/{len(self.image_files)}")
            image = Image.open(file)
            result = self.test_image(image)
            self.resuts.append(result)

        for segmentator in self.resuts:
            segmentator.display_extraction()

    def test_image(self, image):
        segmentator = IrisSegmentator(image)
        # return segmentator.extractor.iris_extracted
        return segmentator
        # segmentator.display_extraction()
        

if __name__ == "__main__":
    tester = Tester()
    tester.test("teczowka_data")
    