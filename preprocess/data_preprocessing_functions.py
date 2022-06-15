import os
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def find_biggest_bounding_box_in_img(path):
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
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img_resize = cv.resize(img, (160, 160), interpolation= cv.INTER_LINEAR)
    img_crop = img_resize[32:32+128, 32:32+128]
    return img_crop

# determine position
def get_crop_position(path, list_dir_32, window_w, window_h):
    x_max = 512
    y_max = 512
    for image_file in list_dir_32:
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

# crop image
def crop_bounding_box_and_resize(img_path, window_w, window_h, img_w, img_h, window_x, window_y): #480 354
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
     # put window around according to position of bounding box
    img_crop = img[window_y:window_y+window_h, window_x:window_x+window_w] # crop window

    img_resize = cv.resize(img_crop, (img_w, img_h), interpolation= cv.INTER_AREA) # resize image .INTER_LINEAR
    return img_resize # return image

def random_down_sampling(list_dir):
    indices = list(range(0, len(list_dir)))
    good_indices = random.sample(indices, 32)
    good_indices.sort()
    list_dir_32 = [list_dir[i] for i in good_indices]
    return list_dir_32

def symmetrical_down_sampling(list_dir):
    m = len(list_dir)
    k = int(m/32)
    if m % 2 == 1:
        idx = int(m / 2) #CHANGE: not start at index 1 but index in the middle
    else:
        idx = int(m / 2)# - 1 #CHANGE: not start at index -1 but at indes in the middle -1
    list_indices = []
    for i in range(32):
        idx = idx + pow((-1), i) * i * k
        list_indices.append(idx)
    list_indices.sort()
    list_dir_32 = [list_dir[i] for i in list_indices]
    return list_dir_32

def stack_2D_images(path, list_dir, sampling):
    result = []
    # delete .ipynb checkpoint
    if list_dir[0] == '.ipynb_checkpoints':
        list_dir.remove('.ipynb_checkpoints')

    if sampling == 'RANDOM':
        list_dir_32 = random_down_sampling(list_dir)
    if sampling == 'SYMMETRIC':
        list_dir_32 = symmetrical_down_sampling(list_dir)

    # here: check x min and y min?
    window_x, window_y = get_crop_position(path, list_dir_32, 480, 384)
    for image_file in list_dir_32:
        if image_file != '.ipynb_checkpoints':

            img_path = os.path.join(path, image_file)

            #image = resize_crop(img_path)
            image = crop_bounding_box_and_resize(img_path, 480, 384, 160, 128, window_x, window_y)
            image = image / 255

            result.append(image)

    image_3D = np.stack(result, axis=0) # 2

    return image_3D

def correct_datasets(dataset_list):
    discard_list = []
    for dataset_idx, dataset in enumerate(dataset_list):
        num_discard = 0
        scan_d_list = []
        for scan_idx, scan in enumerate(dataset):
            discard = False
            w_check = 0
            w_first = 0
            h_first = 0
            for idx, img in enumerate(scan):

                x, y, w, h = find_bounding_box_size(img, 118, 160)

                if idx == 0:
                    w_first = w
                    h_first = h

                if idx == len(scan)-1 and w < w_first and h < h_first and discard == False:
                    discard = True
                    # print(str(number) + ": discard")
                    num_discard +=1

                if idx <= 5 and w < w_check -2 and discard == False:
                    discard = True
                    # print(str(number) + ": discard")
                    num_discard +=1
                w_check = w

                if idx > 8 and idx <= 24 and w < 90 and discard == False:
                    discard = True
                    # print(str(number) + ": discard")
                    num_discard +=1
            if discard:
                scan_d_list.append(scan_idx)
        dataset = np.delete(dataset, scan_d_list, axis=0)
        dataset_list[dataset_idx] = dataset
        discard_list.append(num_discard)
    
    return dataset_list, discard_list