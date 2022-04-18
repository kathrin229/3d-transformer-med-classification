
import os
# import cv2
import cv2
print(cv2.__version__)

# import matplotlib

import matplotlib.pyplot as plt
import matplotlib.image as img

import numpy as np


# TODO Count images per person
# TODO Sample scans with 50 files
# TODO Resize images -> 128x128
# TODO add to matrix

# TODO cut image


# for file in os.listdir(rootdir):
#     d = os.path.join(rootdir, file)
#     print(d)



##### create function
path = '../../data/dataset_seg/CP/1065/3104' #/0022.png'

def stack_2D_images(path):
# first = True
    result = []

    for directory in sorted(os.listdir(path)):
        if directory == '.ipynb_checkpoints':
            break
        img_path = os.path.join(path, directory)
        image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # skip.ipynb_checkpoints
        # take only 50

        down_width = 128
        down_points = (down_width, down_width)
        image_resize = cv2.resize(image_gray, down_points, interpolation= cv2.INTER_LINEAR)

        image_reshaped = np.reshape(image_resize, (128, 128, -1))

        result.append(image_resize)

    image_3D = np.stack(result, axis=2)
    return image_3D

'''
rootpath_CP = '../../data/dataset_seg/CP'

list_CP = []
for directory in sorted(os.listdir(rootpath_CP)):
    subpath = os.path.join(rootpath_CP, directory)
    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)

        image_3D = stack_2D_images(subsubpath)
        
        list_CP.append(image_3D)
        print('checkpoint')


dataset_CP = np.stack(list_CP, axis = 0)
'''



######

    # if first:
    #     result = image_reshaped
    #     first = False
    # else:
    #     result = np.dstack((result, image_reshaped))



# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()

# image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# print(image.shape)
# print(type(image))
# print(type(image.shape))

# down_width = 128
# down_points = (down_width, down_width)
# resize_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
# print(resize_down.shape)

# image = img.imread(path)
# plt.imshow(image)
# plt.show()


# abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
# normal_labels = np.array([0 for _ in range(len(normal_scans))])
# shuffle with train loader

print("done")