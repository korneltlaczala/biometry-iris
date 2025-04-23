import math
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


def binarize(grayscale_image, threshold):
    h, w = grayscale_image.shape[0:2]
    mean_intensity = np.sum(grayscale_image) / (h * w)
    binary_image = np.where(grayscale_image > mean_intensity * threshold, 255, 0).astype(np.uint8)
    
    return binary_image

def keep_largest_contour(image):
    inverted = cv2.bitwise_not(image)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Contours found:", len(contours))
    
    if contours:
        biggest_contour = max(contours, key=cv2.contourArea)
        result = np.ones_like(image) * 255
        cv2.drawContours(result, [biggest_contour], -1, (0), thickness=cv2.FILLED)
        
        return result
    
    return image

def clean_pupil(img):
    image_bin = binarize(img, threshold=0.22)

    # czyszczenie
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image = cv2.erode(image_bin, kernel, iterations=2)
    image = cv2.dilate(image, kernel, iterations=2)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    image = cv2.medianBlur(image, 5)
    image = keep_largest_contour(image)

    return image

def find_pupil_radius(img):
    # projekcja
    binary_image = (img > 0).astype(np.uint8)
    horizontal_proj = np.sum(binary_image, axis=1)
    vertical_proj = np.sum(binary_image, axis=0)

    # środek
    x = int(np.mean(np.where(vertical_proj == np.min(vertical_proj))[0]))
    y = int(np.mean(np.where(horizontal_proj == np.min(horizontal_proj))[0]))

    # promień
    left_edge = np.min(np.where(vertical_proj < max(vertical_proj)))
    right_edge = np.max(np.where(vertical_proj < max(vertical_proj)))
    radius_horizontal = (right_edge - left_edge) // 2

    top_edge = np.min(np.where(horizontal_proj < max(horizontal_proj)))
    bottom_edge = np.max(np.where(horizontal_proj < max(horizontal_proj)))
    radius_vertical = (bottom_edge - top_edge) // 2

    radius_pupil = (radius_horizontal + radius_vertical) // 2

    return int(x), int(y), int(radius_pupil)


def plot_pupil_radius(img, mask):
    x, y, radius = find_pupil_radius(mask)
    # draw a circle on the image
    image_center = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    image_center = cv2.circle(image_center, (x, y), radius, (255, 0, 0), 1)
    image_center = cv2.circle(image_center, (x, y), 0, (255, 0, 0), 5)
    return Image.fromarray(image_center)

def binarize_iris(x, y, r, img):
    mask_pupil = np.zeros_like(img)
    cv2.circle(mask_pupil, (x, y), r, 255, thickness=-1)
    

    mask_outer = np.ones_like(img) * 255
    cv2.circle(mask_outer, (x, y), r, 0, thickness=-1)  

    outer_region = cv2.bitwise_and(img, img, mask=mask_outer)    
    mean_brightness = np.mean(outer_region[mask_outer > 0])

    threshold = mean_brightness / 255 *0.97

    image_bin = binarize(img, threshold=threshold)
   
    
    return image_bin

def extract_iris(iris_mask, pupil_mask):
    iris_mask = cv2.bitwise_xor(iris_mask, pupil_mask)
    iris_mask = cv2.bitwise_not(iris_mask)
    return iris_mask   

def clean_iris(image):
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image = cv2.dilate(image, big_kernel, iterations=1)

    # image = cv2.erode(image, small_kernel, iterations=2)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, small_kernel, iterations=2)
    image = keep_largest_contour(image)
    image = cv2.medianBlur(image, 5)

    return image

def find_iris_radius(img):
    # projekcja
    binary_image = (img > 0).astype(np.uint8)
    horizontal_proj = np.sum(binary_image, axis=1)
    vertical_proj = np.sum(binary_image, axis=0)

    
    # promień
    left_edge = np.min(np.where(vertical_proj < max(vertical_proj)))
    right_edge = np.max(np.where(vertical_proj < max(vertical_proj)))
    radius_horizontal = (right_edge - left_edge) // 2

    top_edge = np.min(np.where(horizontal_proj < max(horizontal_proj)))
    bottom_edge = np.max(np.where(horizontal_proj < max(horizontal_proj)))
    radius_vertical = (bottom_edge - top_edge) // 2

    radius_iris = (radius_horizontal + radius_vertical) // 2

    return int(radius_iris)

def plot_iris_radius(img, mask, x, y):
    radius = find_iris_radius(mask)
    # draw a circle on the image
    image_center = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    image_center = cv2.circle(image_center, (x, y), radius, (255, 0, 0), 1)
    image_center = cv2.circle(image_center, (x, y), 0, (255, 0, 0), 5)
    return Image.fromarray(image_center), radius

import math

def unwrap_iris(img, x, y, r_pupil, r_iris, output_height=64, output_width=360):
    angles = [
        [],                                # dla pasów 0–3
        [(80, 100), (236.5, 303.5)],            # dla pasów 4–5
        [(65, 115), (225, 315)]                     # dla pasów 6–7
    ]

    unwrapped_image = np.zeros((output_height, output_width, img.shape[2] if len(img.shape) == 3 else 1), dtype=img.dtype)
    radius_difference = r_iris - r_pupil

    for v in range(output_height):
        ring_index = int((v / output_height) * 8)
        ring_index = min(ring_index, 7)

        if ring_index <= 3:
            blocked_angles = angles[0]
        elif ring_index <= 5:
            blocked_angles = angles[1]
        else:
            blocked_angles = angles[2]

        for u in range(output_width):
            angle_deg = (u / output_width) * 360
            if any(start <= angle_deg <= end for (start, end) in blocked_angles):
                continue

            angle_rad = math.radians(angle_deg)
            radius = r_pupil + (v / output_height) * radius_difference

            original_x = int(x + radius * math.cos(angle_rad))
            original_y = int(y + radius * math.sin(angle_rad))

            if 0 <= original_y < img.shape[0] and 0 <= original_x < img.shape[1]:
                unwrapped_image[v, u] = img[original_y, original_x]

    return unwrapped_image.squeeze()


def draw_iris_rings(img, x, y, r_pupil, r_iris):
    angles = [
        [],                                # dla pasów 0–3
        [(80, 100), (236.5, 303.5)],            # dla pasów 4–5
        [(65, 115), (225, 315)]                     # dla pasów 6–7
    ]
  
    marked_image = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    radius_difference = r_iris - r_pupil

    for i in range(8):
        radius = r_pupil + (i / 8) * radius_difference
        # Dobranie odpowiednich masek kątowych
        if i <= 3:
            blocked = angles[0]
        elif i <= 5:
            blocked = angles[1]
        else:
            blocked = angles[2]

        # Rysowanie pierścienia w 5-stopniowych wycinkach
        for start in range(0, 360, 5):
            end = start + 5
            is_blocked = any(b_start <= start <= b_end or b_start <= end <= b_end for (b_start, b_end) in blocked)
            color = (255, 0, 0) if is_blocked else (0, 0, 255)  # BGR: blue = blocked, red = normal
            cv2.ellipse(marked_image, (x, y), (int(radius), int(radius)), 0, start, end, color, 1)

    return marked_image



