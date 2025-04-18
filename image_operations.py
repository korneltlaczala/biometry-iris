import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from image_processors import convert_to_grayscale

def compute_histogram(img, sigma=False):
    img_arr = np.array(img)

    if len(img_arr.shape) == 2:
        colors = ['black']
        hist_data = [img_arr]
    else:
        colors = ['red', 'green', 'blue']
        hist_data = [img_arr[:, :, i] for i in range(3)]
    
    plt.figure()
    for data, color in zip(hist_data, colors):
        hist, bins = np.histogram(data.flatten(), bins=256, range=[1, 254])
        # hist, bins = np.histogram(data.flatten(), bins=256, range=[0, 255])
        if sigma:
            hist = gaussian_filter(hist, sigma=2)
        plt.fill_between(bins[:-1], hist, color=color, alpha=0.5)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Image Histogram')
    plt.grid()
    return plt

def horizontal_projection(img):
    img_arr = np.array(img)
    if len(img_arr.shape) == 3:
        img_arr = convert_to_grayscale(img_arr)
        img_arr = np.array(img_arr)
     
    w = len(img_arr[0])
    h = len(img_arr)
    projection = np.zeros(h)
    for i in range(h):
        for j in range(w):
            projection[i] += img_arr[i, j]
    return projection

def vertical_projection(img):
    
    img_arr = np.array(img)

    if len(img_arr.shape) == 3:
        img_arr = convert_to_grayscale(img_arr)
        img_arr = np.array(img_arr)
    
    w = len(img_arr[0])
    h = len(img_arr)
    projection = np.zeros(w)

    for j in range(w):
        for i in range(h):
            projection[j] += img_arr[i, j]
    return projection

def plot_projection(projection, orientation):
    plt.figure()
    
    if orientation == 'Horizontal':
        plt.plot(projection, range(len(projection)))
        plt.ylabel('Row Index')
        plt.xlabel('Pixel Value')
        plt.gca().invert_yaxis()
    else:
        plt.plot(range(len(projection)), projection)
        plt.ylabel('Pixel Value')
        plt.xlabel('Column Index')
    plt.title(f'{orientation} Projection')
    plt.grid()
    return plt

def intersect(img1, img2):
    img1_arr = np.array(img1)
    img2_arr = np.array(img2)
    output_arr = np.zeros(img1_arr.shape)

    if len(img1_arr.shape) == 3 or len(img2_arr.shape) == 3:
        raise ValueError('Both images must be grayscale')
    if img1_arr.shape != img2_arr.shape:
        raise ValueError('Both images must be the same size')
    
    # output_arr = min(img1_arr, img2_arr)
    for i in range(img1_arr.shape[0]):
        for j in range(img1_arr.shape[1]):
            output_arr[i, j] = min(img1_arr[i, j], img2_arr[i, j])
        
    output_img = Image.fromarray(output_arr)
    return output_img

def apply_mask(img, mask):
    img_arr = np.array(img)
    mask_arr = np.array(mask)

    if len(mask_arr.shape) == 2 and len(img_arr.shape) == 3:
        mask_arr = np.expand_dims(mask_arr, axis=-1)
        mask_arr = np.repeat(mask_arr, img_arr.shape[2], axis=-1)

    # output_arr = np.zeros(img_arr.shape)
    # h = img_arr.shape[0]
    # w = img_arr.shape[1]
    # channel_count = img_arr.shape[2] if len(img_arr.shape) == 3 else 1
    # for i in range(h):
    #     for j in range(w):
    #         output_arr[i][j] = img_arr[i][j] * mask_arr[i][j] / 255


    output_arr = (img_arr * mask_arr / 255).astype(np.uint8)
    output_img = Image.fromarray(output_arr.astype(np.uint8))
    return output_img

def get_mean_brightness(img):
    img_arr = np.array(img)
    return np.mean(img_arr)

def erode(img, kernel_size=3, iterations=1):
    img_arr = np.array(img)

    if len(img_arr.shape) == 3:
        raise ValueError('Image must be grayscale')

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(img_arr, kernel, iterations=iterations)
    return Image.fromarray(eroded)

def dilate(img, kernel_size=3, iterations=1):
    img_arr = np.array(img)

    if len(img_arr.shape) == 3:
        raise ValueError('Image must be grayscale')

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(img_arr, kernel, iterations=iterations)
    return Image.fromarray(dilated)

def open(img, kernel_size=3, iterations=1):
    img_arr = np.array(img)

    # if len(img_arr.shape) == 3:
    #     raise ValueError('Image must be grayscale')

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(img_arr, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return Image.fromarray(opened)

def close(img, kernel_size=3, iterations=1):
    img_arr = np.array(img)

    # if len(img_arr.shape) == 3:
    #     raise ValueError('Image must be grayscale')

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(img_arr, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return Image.fromarray(closed)