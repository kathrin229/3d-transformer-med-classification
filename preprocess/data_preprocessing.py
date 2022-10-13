"""
Pre-processing functions for the Clean CC-CCII Dataset https://github.com/wang-shihao/HKBU_HPML_COVID-19
"""

import os
import random
import cv2 as cv
import numpy as np

def find_biggest_bounding_box_in_img(path):
    """
        For exploration: Find biggest bounding box around lung in dataset
        Args:
            path (str): the path to the image
        Returns:
            x (int): x (horizontal) position of bounding box
            y (int): y (vertical) position of bounding box
            w (int): width of bounding box
            h (int): height of bounding box
    """
    im = cv.imread(path)
    img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    ret, binary = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    contours, hierarchy = cv.findContours(
        binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    x = 512
    y = 512
    x_down = 0
    y_down = 0
    for i in range(len(contours)):
        x_new, y_new, w_new, h_new = cv.boundingRect(contours[i])

        if w_new > 100 and h_new > 100:

            x_down_new = x_new + w_new
            y_down_new = y_new + h_new
            if x_new < x:
                x = x_new
            if y_new < y:
                y = y_new
            if x_down_new > x_down:
                x_down = x_down_new
            if y_down_new > y_down:
                y_down = y_down_new

    w = x_down - x
    h = y_down - y
    
    return x, y, w, h

def find_bounding_box_size(img, width, height):
    """
        Detect bounding box in image
        Args:
            img (np.array): the image
            width (int): the width of the image
            height (int): the height of the image
        Returns:
            x (int): x (horizontal) position of bounding box
            y (int): y (vertical) position of bounding box
            w (int): width of bounding box
            h (int): height of bounding box
    """
    img = img*255
    img = img.astype(np.uint8)
    ret, binary = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    contours, hierarchy = cv.findContours(
        binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    x = width
    y = height
    x_down = 0
    y_down = 0
    for i in range(len(contours)):
        x_new, y_new, w_new, h_new = cv.boundingRect(contours[i])

        x_down_new = x_new + w_new
        y_down_new = y_new + h_new
        if x_new < x:
            x = x_new
        if y_new < y:
            y = y_new
        if x_down_new > x_down:
            x_down = x_down_new
        if y_down_new > y_down:
            y_down = y_down_new

    w = x_down - x
    h = y_down - y
    
    return x, y, w, h

def resize_crop(img_path):
    """
        Resize image and crop (technique from paper https://www.medrxiv.org/content/10.1101/2020.06.08.20125963v2)
        Args:
            img_path (str): path to the image to resize and crop
        Returns:
            img_crop (np.array): resized and cropped image
    """
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img_resize = cv.resize(img, (160, 160), interpolation= cv.INTER_AREA)
    img_crop = img_resize[32:32+128, 32:32+128]
    return img_crop


def get_crop_position(path, list_dir_small, window_w, window_h):
    """
        Determine position to crop scan for different crop and resize method with bounding box.
        All images of one scan should be cropped at the same position to maintain the physiological properties of the 3D lung.
        Args:
            path (str): path to the image to resize and crop
            list_dir_small (str): directory leading to the folder where all scan images are located
            window_w (int): width of bounding box window to fit lung in
            window_h (int): height of bounding box window to fit lung in
        Returns:
            x_max (int): x position of window to crop all scan images
            y_max (int): y position of window to crop all scan images
    """
    x_max = 512
    y_max = 512
    for image_file in list_dir_small:
        if image_file != '.ipynb_checkpoints':
            img_path = os.path.join(path, image_file)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            x, y, w, h = find_biggest_bounding_box_in_img(img_path)
            if w > 0 and w <= window_w and h <= window_h:
                window_x = x
                window_y = y

                if window_x + window_w > 512 and window_y + window_h > 512:
                    window_x = window_x - (window_w - w)
                    window_y = window_y - (window_h - h)
                elif window_x + window_w > 512:
                    window_x = window_x - (window_w - w)
                elif window_y + window_h > 512:
                    window_y = window_y - (window_h - h)

                if window_x < 0 or window_y < 0:
                    window_x = int((512 - window_w) /2)
                    window_y = int((512 - window_h) /2)
                
                if window_x < x_max:
                    x_max = window_x
                if window_y < y_max:
                    y_max = window_y
                    
    if x_max == 512 or y_max == 512:
        x_max = int((512 - window_w) /2)
        y_max = int((512 - window_h) /2)
    return x_max, y_max


def crop_bounding_box_and_resize(img_path, window_w, window_h, img_w, img_h, window_x, window_y): #480 354
    """
        Crops image at bounding box and resizes image (new strategy)
        Args:
            img_path (str): path to the image to resize and crop
            window_w (int): width of bounding box window
            window_h (int): height of bounding box window
            img_w (int): width of the image
            img_h (int): height of the image
            window_x (int): x (horizontal) position of where to place bounding box
            window_y (int): y (vertical) position of where to place bounding box
        Returns:
            img_resize (np.array): resized image
    """
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    # put window around according to position of bounding box, crop window
    img_crop = img[window_y:window_y+window_h, window_x:window_x+window_w]
    # resize image
    img_resize = cv.resize(img_crop, (img_w, img_h), interpolation= cv.INTER_AREA) 

    return img_resize

def random_down_sampling(list_dir, num_samples):
    """
        Paper strategy 1 (not used): choose random images from scan to sample scan
        Args:
            list_dir ([str]): list of directories
            num_samples (int): number of images to be sampled
        Returns:
            list_dir_small ([str]): sampled list of directories
    """
    indices = list(range(0, len(list_dir)))
    good_indices = random.sample(indices, num_samples)
    good_indices.sort()
    list_dir_small = [list_dir[i] for i in good_indices]
    return list_dir_small

def symmetrical_down_sampling(list_dir, num_samples):
    """
        Paper strategy 2 (used). Corrected version of strategy described in pseudocode
        Args:
            list_dir ([str]): list of directories
            num_samples (int): number of images to be sampled
        Returns:
            list_dir_small ([str]): sampled list of directories
    """
    m = len(list_dir)
    k = int(m/num_samples)
    if m % 2 == 1:
        idx = int(m / 2) #CHANGE: not start at index 1 but index in the middle
    else:
        idx = int(m / 2)# - 1 #CHANGE: not start at index -1 but at indes in the middle -1
    list_indices = []
    for i in range(num_samples):
        idx = idx + pow((-1), i) * i * k
        list_indices.append(idx)
    list_indices.sort()
    list_dir_small = [list_dir[i] for i in list_indices]
    return list_dir_small

def stack_2D_images(path, list_dir, sampling, num_samples, window_w, window_h, img_w, img_h):
    """
        Creates 3D ct scans by using sampling and cropping methods
        Args:
            path (str): path to the scan to stack images together
            list_dir ([str]): a list containing all image file names of the scan
            sampling (str): 'RANDOM' or 'SYMMETRIC' sampling
            num_samples (int): number of samples to get from every scan
            window_w (int): width of bounding box window
            window_h (int): height of bounding box window
            img_w (int): width of the image
            img_h (int): height of the image
        Returns:
            img_3D (np.array): 3D numpy array containing all images of one scan
    """
    result = []
    # delete .ipynb checkpoint
    if list_dir[0] == '.ipynb_checkpoints':
        list_dir.remove('.ipynb_checkpoints')

    if sampling == 'RANDOM':
        list_dir_small = random_down_sampling(list_dir, num_samples)
    if sampling == 'SYMMETRIC':
        list_dir_small = symmetrical_down_sampling(list_dir, num_samples)

    # here: check x min and y min?
    window_x, window_y = get_crop_position(path, list_dir_small, window_w, window_h)
    for image_file in list_dir_small:
        if image_file != '.ipynb_checkpoints':

            img_path = os.path.join(path, image_file)

            #image = resize_crop(img_path)
            image = crop_bounding_box_and_resize(img_path, window_w, window_h, img_w, img_h, window_x, window_y)
            image = image / 255

            result.append(image)

    image_3D = np.stack(result, axis=0) # 2

    return image_3D

def correct_datasets(dataset_list, img_w, img_h):
    """
        Detects faulty scans in the created dataset
        Args:
            dataset_list ([np.array]): list of 3D arrays (images)
            img_w (int): width of the image
            img_h (int): height of the image
        Returns:
            img_3D (np.array): 3D numpy array containing all images of one scan
    """
    discard_list = []
    threshold = 2
    for dataset_idx, dataset in enumerate(dataset_list):
        num_discard = 0
        scan_d_list = []
        for scan_idx, scan in enumerate(dataset):
            discard = False
            w_check = 0
            w_first = 0
            h_first = 0
            for idx, img in enumerate(scan):

                x, y, w, h = find_bounding_box_size(img, img_w, img_h)

                if idx == 0:
                    w_first = w
                    h_first = h

                # if box in last slice smaller than box in first slice: lung is reverse, discard
                if idx == len(scan)-1 and w < w_first and h < h_first and discard == False:
                    discard = True
                    num_discard +=1

                # increasing during the first quater of the lung: 32/6
                if idx < int(len(scan)/5) and w < w_check - threshold and discard == False:
                    discard = True
                    num_discard +=1
                w_check = w

                # slice similar to beginning's slices width in the middle
                if idx > int(len(scan)/4) and idx < len(scan) - int(len(scan)/4) and w < int(img_h/1.7) and discard == False:
                    discard = True
                    num_discard +=1
            if discard:
                scan_d_list.append(scan_idx)
        dataset = np.delete(dataset, scan_d_list, axis=0)
        dataset_list[dataset_idx] = dataset
        discard_list.append(num_discard)
    
    return dataset_list, discard_list