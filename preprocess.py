
import os
# import cv2
import cv2
print(cv2.__version__)

# import matplotlib

# import matplotlib.pyplot as plt
# import matplotlib.image as img

import numpy as np

# import tarfile
  
# file = tarfile.open('dataset_seg.tar.gz')
# file.extractall('./data')
  
# file.close()

# print('extracted')


# Done Count images (per lung) and do statistics --> playground "create table overview + graphs"
# Done RUN Sample scans, resize, normalize add to dataset, save for all
# Done start model building

# TODO sampling strategy
# TODO cut image
# TODO improve CNN


# for file in os.listdir(rootdir):
#     d = os.path.join(rootdir, file)
#     print(d)



##### create function
# path = '../../data/dataset_seg/CP/1065/3104' #/0022.png'

def stack_2D_images(path):
# first = True
    result = []
    list_dir = sorted(os.listdir(path))
    
    start = int((len(list_dir) - 50) /2)
    list_dir_50 = list_dir[start:start+50]


    for image_file in list_dir_50:
        if image_file == '.ipynb_checkpoints':
            break
        img_path = os.path.join(path, image_file)
        image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # skip.ipynb_checkpoints
        # take only 50

        down_width = 128
        down_points = (down_width, down_width)
        image_resize = cv2.resize(image_gray, down_points, interpolation= cv2.INTER_LINEAR)
        image_resize = image_resize / 255

        # image_reshaped = np.reshape(image_resize, (128, 128, -1)) TODO for 3D CNNs

        result.append(image_resize)

    image_3D = np.stack(result, axis=0) # 2
    return image_3D


rootpath_CP = './data/dataset_seg/CP'
rootpath_NCP = './data/dataset_seg/NCP'
rootpath_Normal = './data/dataset_seg/Normal'

list_CP = []
list_NCP = []
list_Normal = []

for directory in sorted(os.listdir(rootpath_CP)):
    subpath = os.path.join(rootpath_CP, directory)

    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)

        if len(os.listdir(subsubpath)) > 50:
            image_3D = stack_2D_images(subsubpath)
            list_CP.append(image_3D)
print('CP done')

for directory in sorted(os.listdir(rootpath_NCP)):
    subpath = os.path.join(rootpath_NCP, directory)

    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)

        if len(os.listdir(subsubpath)) > 50:
            image_3D = stack_2D_images(subsubpath)
            list_NCP.append(image_3D)

print('NCP done')

for directory in sorted(os.listdir(rootpath_Normal)):
    subpath = os.path.join(rootpath_Normal, directory)

    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)

        if len(os.listdir(subsubpath)) > 50:
            image_3D = stack_2D_images(subsubpath)
            list_Normal.append(image_3D)
print('Normal done')

dataset_CP = np.stack(list_CP, axis = 0)
dataset_NCP = np.stack(list_NCP, axis = 0)
dataset_Normal = np.stack(list_Normal, axis = 0)

np.savez_compressed('data-arrays/dataset_CP', dataset_CP)
np.savez_compressed('data-arrays/dataset_NCP', dataset_NCP)
np.savez_compressed('data-arrays/dataset_Normal', dataset_Normal)

print('done')




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