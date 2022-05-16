import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# TODO function that returns distance to border
# TODO calculate minimum distances to all borders -> change of plans


##############################################################
#path = 'data/dataset_seg/CP/1/3147/0006.png'

def delete_empty_image(path):
    im = cv.imread(path)
    img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    ret, binary = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    contours, hierarchy = cv.findContours(
        binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        os.remove(path)


def find_biggest_bounding_box_in_img(path):
    im = cv.imread(path)
    img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    ret, binary = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    contours, hierarchy = cv.findContours(
        binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #print("Number of contours:" + str(len(contours)))
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

    cv.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 3)
    plt.imshow(im)
    plt.show()
    
    return x, y, w, h

# data/dataset_seg/NCP/1042/2614/0000.png
# find_biggest_bounding_box_in_img('./data/dataset_seg/CP/1231/3449/0245.png')
#'./data/dataset_seg/NCP/2715/2708/0005.png'
#'./data/dataset_seg/NCP/913/2455/0181.png'

#'./data/dataset_seg/CP/2439/2909/0333.png'
#'./data/dataset_seg/CP/15/3174/0097.png'

# FINAL:
#'./data/dataset_seg/NCP/2715/2708/0005.png'
#'./data/dataset_seg/CP/1231/3449/0245.png'

#'./data/dataset_seg/Normal/790/225/0025.png'

###########################################################

'''
rootpath = './data/dataset_seg'
labels = ['CP', 'NCP', 'Normal']

img_size = 512

w_min = 0
h_min = 0

path_min_w = ''
path_min_h = ''

for label in labels:
    rootpath_label = rootpath + '/' + label

    for directory in sorted(os.listdir(rootpath_label)):
        subpath = os.path.join(rootpath_label, directory)

        if subpath == './data/dataset_seg/CP/1357':
            break
        if subpath == './data/dataset_seg/CP/1255':
            break
        

        for subdirectory in sorted(os.listdir(subpath)):
            subsubpath = os.path.join(subpath, subdirectory)

            if subsubpath == './data/dataset_seg/NCP/913/2455':#/0181.png':
                break

            # if subsubpath == './data/dataset_seg/CP/2439/2909/0333.png':
            #     break

            # if subsubpath == './data/dataset_seg/CP/15/3174/0097.png':
            #     break

            # './data/dataset_seg/NCP/1042/2614'
            # './data/dataset_seg/NCP/1043/2615'
            #'./data/dataset_seg/NCP/125/1394'

            list_dir = sorted(os.listdir(subsubpath))

            for image_file in list_dir:
                if image_file == '.ipynb_checkpoints':
                    break
                img_path = os.path.join(subsubpath, image_file)

                if img_path == './data/dataset_seg/NCP/913/2455/0181.png':
                    break

                if img_path == './data/dataset_seg/CP/2439/2909/0333.png':
                    break

                if img_path == './data/dataset_seg/CP/15/3174/0097.png':
                    break
                # im = cv.imread(img_path)
                # img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                x, y, w, h = find_biggest_bounding_box_in_img(img_path)

                # if w > 500:
                #     break
                    #'./data/dataset_seg/CP/1357/3759/0004.png'

                if w > w_min:
                    w_min = w
                    path_min_w = img_path

                if h > h_min:
                    h_min = h
                    path_min_h = img_path
                # then: save img path with biggest lung width and height
    print('label ' + label + ' done')
            
print(str(w_min))
print(str(h_min))



rootpath = './data/dataset_seg'
labels = ['CP', 'NCP', 'Normal']

for label in labels:
    rootpath_label = rootpath + '/' + label

    for directory in sorted(os.listdir(rootpath_label)):
        subpath = os.path.join(rootpath_label, directory)

        for subdirectory in sorted(os.listdir(subpath)):
            subsubpath = os.path.join(subpath, subdirectory)

            list_dir = sorted(os.listdir(subsubpath))

            for image_file in list_dir:
                if image_file == '.ipynb_checkpoints':
                    break
                img_path = os.path.join(subsubpath, image_file)
                delete_empty_image(img_path)

    print('label ' + label + ' done')
'''
#'./data/dataset_seg/NCP/2715/2708/0005.png'
#'./data/dataset_seg/Normal/790/225/0025.png'
# print('done')
