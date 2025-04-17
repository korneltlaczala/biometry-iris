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