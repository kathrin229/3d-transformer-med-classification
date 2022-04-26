import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# TODO function that returns distance to border
# TODO calculate minimum distances to all borders



##############################################################
#path = 'data/dataset_seg/CP/1/3147/0006.png'


def find_biggest_bounding_box(img):
    # im = cv.imread(path)
    # img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    ret,binary = cv.threshold(img,127,255,cv.THRESH_TRUNC)
    contours,hierarchy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    #print("Number of contours:" + str(len(contours)))
    x = 512
    y = 512
    x_down = 0
    y_down = 0
    for i in range(len(contours)):
        x_new,y_new,w_new,h_new = cv.boundingRect(contours[i])
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

    w = x_down -x
    h = y_down -y

    return x, y, w, h
# cv.rectangle(im,(x,y),(x+w,y+h),(255,0,0),3)


# plt.imshow(im)
# plt.show()
###########################################################


rootpath_CP = './data/dataset_seg/Normal'

img_size = 512

count_too_big = 0

up_min = img_size
down_min = img_size
left_min = img_size
right_min = img_size

for directory in sorted(os.listdir(rootpath_CP)):
    subpath = os.path.join(rootpath_CP, directory)

    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)

        list_dir = sorted(os.listdir(subsubpath))

        first_file = True

        for image_file in list_dir:
            if image_file == '.ipynb_checkpoints':
                break
            img_path = os.path.join(subsubpath, image_file)

            im = cv.imread(img_path)
            img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

            if img.shape[0] > 512:
                if first_file:
                    count_too_big += 1
                    first_file = False
                break

            x, y, w, h = find_biggest_bounding_box(img)

            if w<0:
                break

            if x < left_min:
                left_min = x
            if y < up_min:
                up_min = y
            if img_size - (x + w) < right_min:
                right_min = img_size - (x + w)
            if img_size - (y + h) < down_min:
                down_min = img_size - (y + h)

            if(down_min < 0):
                print('crap')

            if(right_min < 0):
                print('crap')
            
print('CP done')

print(str(up_min))
print(str(down_min))
print(str(left_min))
print(str(right_min))


print('done')
