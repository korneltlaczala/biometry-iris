import numpy as np
from PIL import Image

def convert_to_grayscale(img):
    grayscale_processor = GrayscaleProcessor()
    grayscale_processor.set_param("_is_enabled", True)
    return grayscale_processor.process(img)

class ProcessorFlow:
    def __init__(self):
        self.processors = []
        self.last_run_changed_img = True

    def add_processor(self, processor):
        self.processors.append(processor)

    def process(self, img, pixelwise=False):
        self.last_run_changed_img = False
        changed_params = False
        for processor in self.processors:
            if processor.changed_params:
                changed_params = True
            if not changed_params:
                img = processor.get_last_img()
                continue
            img = processor.process(img, pixelwise=pixelwise)
            self.last_run_changed_img = True
        return img

    def reset_cache(self):
        for processor in self.processors:
            processor.changed_params = True

    def reset(self):
        for processor in self.processors:
            processor.set_default_params()
        self.reset_cache()

    def __str__(self):
        output = "-" * 20 + "\n"
        output += "ProcessorFlow:\n"
        output += "-" * 20 + "\n"
        for processor in self.processors:
            output += repr(processor) + "\n"
        output += "-" * 20 + "\n"
        return output


class Processor:
    def __init__(self):
        self.changed_params = True
        self.last_img = None

    def process(self, img, pixelwise=False):
        self.changed_params = False
        img_arr = np.array(img, dtype=np.int32)
        if pixelwise:
            img_arr = self._process_pixelwise(img_arr).astype(np.uint8)
        else:
            img_arr = self._process(img_arr).astype(np.uint8)
        self.last_img = Image.fromarray(img_arr)
        return self.last_img

    def _process(self, img_arr):
        raise NotImplementedError

    def _process_pixelwise(self, img_arr):
        raise NotImplementedError

    def set_param(self, param_name, value):
        if getattr(self, param_name) == value:
            return
        self.changed_params = True
        setattr(self, param_name, value)

    def get_last_img(self):
        return self.last_img

    def set_default_params(self):
        raise NotImplementedError
        
class BrightnessProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_value = 0
        self._value = self.default_value

    def _process(self, img_arr):
        img_arr = np.clip(img_arr + self._value, 0, 255)
        return img_arr

    def _process_pixelwise(self, img_arr):  
        if len(img_arr.shape) == 2:
            for i in range(img_arr.shape[0]):
                for j in range(img_arr.shape[1]):
                    img_arr[i, j] = np.clip(img_arr[i, j] + self._value, 0, 255)
            return img_arr

        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for c in range(img_arr.shape[2]):
                    img_arr[i, j, c] = np.clip(img_arr[i, j, c] + self._value, 0, 255)
        return img_arr

    @property
    def value(self):
        return self._value

    def set_default_params(self):
        self._value = self.default_value

    def __repr__(self):
        return f"BrightnessProcessor(value={self._value})"

class ExposureProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_factor = 1.0
        self._factor = self.default_factor

    def _process(self, img_arr):
        img_arr = np.clip(img_arr * self._factor, 0, 255)
        return img_arr

    def _process_pixelwise(self, img_arr):  

        if len(img_arr.shape) == 2:
            for i in range(img_arr.shape[0]):
                for j in range(img_arr.shape[1]):
                    img_arr[i, j] = np.clip(img_arr[i, j] * self._factor, 0, 255)
            return img_arr

        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for c in range(img_arr.shape[2]):
                    img_arr[i, j, c] = np.clip(img_arr[i, j, c] * self._factor, 0, 255)
        
        return img_arr

    @property
    def factor(self):
        return self._factor

    def set_default_params(self):
        self._factor = self.default_factor

    def __repr__(self):
        return f"ExposureProcessor(factor={self._factor})"


class ContrastProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_factor = 1.0
        self._factor = self.default_factor

    def _process(self, img_arr):
        mean = np.mean(img_arr)
        img_arr = np.clip((img_arr - mean) * self._factor + mean, 0, 255)
        return img_arr

    def _process_pixelwise(self, img_arr):  

        sum = 0
        if len(img_arr.shape) == 2:
            for i in range(img_arr.shape[0]):
                for j in range(img_arr.shape[1]):
                    sum += img_arr[i, j]
            mean = sum / img_arr.shape[0] / img_arr.shape[1]

            for i in range(img_arr.shape[0]):
                for j in range(img_arr.shape[1]):
                    img_arr[i, j] = np.clip((img_arr[i, j] - mean) * self._factor + mean, 0, 255)
            return img_arr

        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for c in range(img_arr.shape[2]):
                    sum += img_arr[i, j, c]
        mean = sum / img_arr.shape[0] / img_arr.shape[1] / img_arr.shape[2]

        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for c in range(img_arr.shape[2]):
                    img_arr[i, j, c] = np.clip((img_arr[i, j, c] - mean) * self._factor + mean, 0, 255)
        return img_arr
    
    @property
    def factor(self):
        return self._factor

    def set_default_params(self):
        self._factor = self.default_factor
    
    def __repr__(self):
        return f"ContrastProcessor(factor={self._factor})"


class GammaProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_factor = 1.0
        self._factor = self.default_factor

    def _process(self, img_arr):
        img_arr = np.power(img_arr / 255.0, self._factor) * 255
        return img_arr

    def _process_pixelwise(self, img_arr):  
        if len(img_arr.shape) == 2:
            for i in range(img_arr.shape[0]):
                for j in range(img_arr.shape[1]):
                    img_arr[i, j] = np.power(img_arr[i, j] / 255.0, self._factor) * 255
            return img_arr
        
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for c in range(img_arr.shape[2]):
                    img_arr[i, j, c] = np.power(img_arr[i, j, c] / 255.0, self._factor) * 255
        return img_arr
    
    @property
    def factor(self):
        return self._factor

    def set_default_params(self):
        self._factor = self.default_factor
    
    def __repr__(self):
        return f"GammaProcessor(factor={self._factor})"


class GrayscaleProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.default_is_enabled = False
        self._is_enabled = self.default_is_enabled
         
    def _process(self, img_arr):
        if not self._is_enabled:
            return img_arr
        if len(img_arr.shape) == 2:
            return img_arr
        img_arr = np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140])
        return img_arr

    def _process_pixelwise(self, img_arr):
        if not self._is_enabled:
            return img_arr
        if len(img_arr.shape) == 2:
            return img_arr

        new_img_arr = np.zeros((img_arr.shape[0], img_arr.shape[1]))
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                value = np.dot(img_arr[i][j], [0.2989, 0.5870, 0.1140])
                new_img_arr[i][j] = value
        return new_img_arr

    @property
    def is_enabled(self):
        return self._is_enabled

    def set_default_params(self):
        self._is_enabled = self.default_is_enabled

    def __repr__(self):
        return f"GrayscaleProcessor(is_enabled={self._is_enabled})"


class NegativeProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_is_enabled = False
        self._is_enabled = self.default_is_enabled

    def _process(self, img_arr):
        if not self._is_enabled:
            return img_arr
        img_arr = 255 - img_arr
        return img_arr
    
    def _process_pixelwise(self, img_arr):  
        if not self._is_enabled:
            return img_arr

        if len(img_arr.shape) == 2:
            for i in range(img_arr.shape[0]):
                for j in range(img_arr.shape[1]):
                    img_arr[i][j] = 255 - img_arr[i][j]
            return img_arr

        channels = 3
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for c in range(channels):
                    img_arr[i, j, c] = 255 - img_arr[i, j, c]
        return img_arr

    @property
    def is_enabled(self):
        return self._is_enabled

    def set_default_params(self):
        self._is_enabled = self.default_is_enabled

    def __repr__(self):
        return f"NegativeProcessor(is_enabled={self._is_enabled})"


class BinarizationProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_is_enabled = False
        self._is_enabled = self.default_is_enabled
        self.default_threshold = 128
        self._threshold = self.default_threshold

    def _process(self, img_arr):
        if not self._is_enabled:
            return img_arr
        img_arr = np.where(img_arr > self._threshold, 255, 0)
        return img_arr

    def _process_pixelwise(self, img_arr):  
        if not self._is_enabled:
            return img_arr
        if len(img_arr.shape) == 2:
            for i in range(img_arr.shape[0]):
                for j in range(img_arr.shape[1]):
                    img_arr[i][j] = np.where(img_arr[i][j] > self._threshold, 255, 0)
            return img_arr
        
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for c in range(channels):
                    img_arr[i, j, c] = np.where(img_arr[i, j, c] > self._threshold, 255, 0)
        return img_arr
    
    @property
    def is_enabled(self):
        return self._is_enabled

    @property
    def threshold(self):
        return self._threshold

    def set_default_params(self):
        self._is_enabled = self.default_is_enabled
        self._threshold = self.default_threshold

    def __repr__(self):
        return f"BinarizationProcessor(is_enabled={self._is_enabled}, threshold={self._threshold})"


class FilterProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.default_is_enabled = False
        self._is_enabled = self.default_is_enabled
        self.default_size = 3
        self._size = self.default_size

    @property
    def size(self):
        return self._size

    def set_default_params(self):
        self._is_enabled = self.default_is_enabled
        self._size = self.default_size

    def __repr__(self):
        return f"FilterProcessor(is_enabled={self._is_enabled}, size={self._size})"


class MeanFilterProcessor(FilterProcessor):

    def _process(self, img_arr):
        if not self._is_enabled:
            return img_arr

        kernel = MeanKernel(self.size)
        img_arr = kernel.convolute(img_arr)
        return img_arr

    def _process_pixelwise(self, img_arr):
        return self._process(img_arr)

    def __repr__(self):
        return f"MeanFilterProcessor(is_enabled={self._is_enabled}, size={self._size})"


class GaussianFilterProcessor(FilterProcessor):

    def __init__(self):
        super().__init__()
        self.default_sigma = 1.0
        self._sigma = self.default_sigma

    def _process(self, img_arr):
        if not self._is_enabled:
            return img_arr

        kernel = GaussianBlurKernel(self.size, self.sigma)
        img_arr = kernel.convolute(img_arr)
        return img_arr

    def _process_pixelwise(self, img_arr):
        return self._process(img_arr)

    @property
    def sigma(self):
        return self._sigma

    def set_default_params(self):
        super().set_default_params()
        self._sigma = self.default_sigma

    def __repr__(self):
        return f"GaussianFilterProcessor(is_enabled={self._is_enabled}, size={self._size}, sigma={self._sigma})"


class SharpeningFilterProcessor(FilterProcessor):

    def __init__(self):
        super().__init__()
        self.default_strength = 1.0
        self._strength = self.default_strength
        self.default_type = "basic"
        self._type = self.default_type

    def _process(self, img_arr):
        if not self._is_enabled:
            return img_arr

        if self.type == "basic":
            kernel = SharpeningKernel(self.size, self.strength)
        elif self.type == "strong":
            kernel = StrongSharpeningKernel(self.size, self.strength)
        img_arr = kernel.convolute(img_arr)
        return img_arr

    def _process_pixelwise(self, img_arr):
        return self._process(img_arr)
    
    @property
    def strength(self):
        return self._strength
    
    @property
    def type(self):
        return self._type

    def set_default_params(self):
        super().set_default_params()
        self._strength = self.default_strength
        self._type = self.default_type

    def __repr__(self):
        return f"SharpeningFilterProcessor(is_enabled={self._is_enabled}, size={self._size}, strength={self._strength}, type={self._type})"



class EdgeDetectionProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.computation_needed = True

    def process(self, img, threshold):
        if not self.computation_needed:
            return self.apply_threshhold(self.last_img, threshold)

        img = convert_to_grayscale(img)
        img_arr = np.array(img, dtype=np.int32)
        img_arr = self._process(img_arr).astype(np.uint8)
        self.last_img = Image.fromarray(img_arr)
        processed_img = self.apply_threshhold(self.last_img, threshold)

        self.computation_needed = False
        return processed_img

    def apply_threshhold(self, img, threshold):
        if threshold == -1:
            return img
        img_arr = np.array(img, dtype=np.int32)
        img_arr[img_arr < threshold] = 0
        img_arr[img_arr >= threshold] = 255
        return Image.fromarray(img_arr.astype(np.uint8))

    def _compute_edge_detection(self, img_arr):
        Gx = self.XKernel.convolute(img_arr)
        Gy = self.YKernel.convolute(img_arr)

        magnitude = np.sqrt(Gx**2 + Gy**2)
        magnitude = magnitude / np.max(magnitude) * 255
        # magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude)) * 255
        return magnitude

class RobertsCrossProcessor(EdgeDetectionProcessor):
    def _process(self, img_arr):
        self.XKernel = XRobertsCrossKernel()
        self.YKernel = YRobertsCrossKernel()
        return self._compute_edge_detection(img_arr)
        
class SobelOperatorProcessor(EdgeDetectionProcessor):
    def _process(self, img_arr):
        self.XKernel = XSobelOperatorKernel()
        self.YKernel = YSobelOperatorKernel()
        return self._compute_edge_detection(img_arr)


class Kernel():

    def __init__(self):
        self.kernel = None

    def convolute(self, img_arr):
        h, w = img_arr.shape[0], img_arr.shape[1]
        pad = self.kernel.shape[0] // 2

        result = np.zeros_like(img_arr)

        if len(img_arr.shape) == 3:
            chanels = img_arr.shape[2]
            grayscale = False
            img_padded = np.pad(img_arr, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        else:
            grayscale = True
            img_padded = np.pad(img_arr, ((pad, pad), (pad, pad)), mode='reflect')


        for i in range(h):
            for j in range(w):
                if grayscale:
                    window = img_padded[i:i + self.kernel.shape[0], j:j + self.kernel.shape[0]]
                    result[i, j] = np.sum(window * self.kernel)
                    continue
                for c in range(chanels):
                    window = img_padded[i:i + self.kernel.shape[0], j:j + self.kernel.shape[0], c]
                    result[i, j, c] = np.sum(window * self.kernel)
        
        result = np.clip(result, 0, 255)
        return result


class MeanKernel(Kernel):
    def __init__(self, size):
        self.kernel = np.ones((size, size))
        self.kernel = self.kernel / np.sum(self.kernel)


class GaussianBlurKernel(Kernel):
    def __init__(self, size, sigma):
        self.kernel = np.fromfunction(
            lambda x, y: (1/ (2 * np.pi * sigma **2)) * np.exp(-((x - (size - 1)/2)**2 + (y - (size - 1)/2)**2) / (2 * sigma ** 2)),
            (size, size)
        )
        self.kernel = self.kernel / np.sum(self.kernel)

class SharpeningKernel(Kernel):
    def __init__(self, size, strength):
        mid = size // 2
        self.kernel = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                dist = abs(i - mid) + abs(j - mid)
                self.kernel[i, j] = min(0, dist - mid - 1)
        
        desired_sum = 4*strength
        self.kernel[mid, mid] = 0
        self.kernel[mid, mid] = -np.sum(self.kernel)
        scale = desired_sum / self.kernel[mid, mid]
        self.kernel = self.kernel * scale
        self.kernel[mid, mid] += 1

class StrongSharpeningKernel(Kernel):
    def __init__(self, size, strength):
        mid = size // 2
        self.kernel = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                dist = max(abs(i - mid), abs(j - mid))
                self.kernel[i, j] = min(0, dist - mid - 1)
        
        desired_sum = 4*strength
        self.kernel[mid, mid] = 0
        self.kernel[mid, mid] = -np.sum(self.kernel)
        scale = desired_sum / self.kernel[mid, mid]
        self.kernel = self.kernel * scale
        self.kernel[mid, mid] += 1

class XRobertsCrossKernel(Kernel):
    def __init__(self):
        self.kernel = np.array([[1, 0], [0, -1]])

class YRobertsCrossKernel(Kernel):
    def __init__(self):
        self.kernel = np.array([[0, 1], [-1, 0]])

class XSobelOperatorKernel(Kernel):
    def __init__(self):
        self.kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

class YSobelOperatorKernel(Kernel):
    def __init__(self):
        self.kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])