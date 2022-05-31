import os
import cv2
from bounding_box import find_biggest_bounding_box_in_img

def resize_crop(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img, (160, 160), interpolation= cv2.INTER_LINEAR)
    img_crop = img_resize[32:32+128, 32:32+128]
    return img_crop

# TODO new function that first determines the x and y position of the lung


def crop_bounding_box_and_resize(img_path, window_w, window_h): #480 354
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # get biggest bounding box in img
    x, y, w, h = find_biggest_bounding_box_in_img(img_path)
    # case no box found
    if w < 0:
        # center crop
        window_x = int((512 - window_w) /2)
        window_y = int((512 - window_h) /2)
        img_crop = img[window_y:window_y+window_h, window_x:window_x+window_w]
    # case box bigger than window w and h
    elif w > window_w or h > window_h:
        # center crop
        window_x = int((512 - window_w) /2)
        window_y = int((512 - window_h) /2)
        img_crop = img[window_y:window_y+window_h, window_x:window_x+window_w]
    else:
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
        

        # put window around according to position of bounding box
        img_crop = img[window_y:window_y+window_h, window_x:window_x+window_w] # crop window

    img_resize = cv2.resize(img_crop, (160, 118), interpolation= cv2.INTER_LINEAR) # resize image
    return img_resize # return image

def resize_all_images_512():
    rootpath = './data/dataset_seg'
    labels = ['CP', 'NCP', 'Normal']

    img_size = 512

    for label in labels:
        rootpath_label = rootpath + '/' + label

        for directory in sorted(os.listdir(rootpath_label)):
            subpath = os.path.join(rootpath_label, directory)

            for subdirectory in sorted(os.listdir(subpath)):
                subsubpath = os.path.join(subpath, subdirectory)

                for image_file in sorted(os.listdir(subsubpath)):
                    if image_file == '.ipynb_checkpoints':
                        break
                    img_path = os.path.join(subsubpath, image_file)

                    im = cv.imread(img_path)
                    img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

                    if img.shape[0] != img_size:
                        down_width = img_size
                        img_resize = cv.resize(img, (down_width, down_width), interpolation= cv.INTER_LINEAR)
                        cv.imwrite(img_path, img_resize) 

