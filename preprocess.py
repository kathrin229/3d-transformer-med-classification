import os
import cv2
import random
import numpy as np

from sampling import random_down_sampling
from sampling import symmetrical_down_sampling

from resize import resize_crop
from resize import crop_bounding_box_and_resize

# Done Count images (per lung) and do statistics --> playground "create table overview + graphs"
# Done RUN Sample scans, resize, normalize add to dataset, save for all
# Done start model building
# Done resize images to 512
# Done figure out max bounding box size
# Done split train - test 
# Done new functions with sampling strategy
# Done cut image according to results of bounding box

# TODO create new dataset
# TODO improve CNN


def stack_2D_images(path, sampling):
# first = True
    result = []
    list_dir = sorted(os.listdir(path))

    # delete .ipynb checkpoint
    if list_dir[0] == '.ipynb_checkpoints':
        list_dir.remove('.ipynb_checkpoints')

    if sampling == 'RANDOM':
        list_dir_32 = random_down_sampling(list_dir)
    if sampling == 'SYMMETRIC':
        list_dir_32 = symmetrical_down_sampling(list_dir)

    # start = int((len(list_dir) - 50) /2)
    # list_dir_50 = list_dir[start:start+50]


    for image_file in list_dir_32:
        if image_file == '.ipynb_checkpoints':
            break
        img_path = os.path.join(path, image_file)

        #image = resize_crop(img_path)
        image = crop_bounding_box_and_resize(img_path, 477, 354)

        # image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # skip.ipynb_checkpoints
        # # take only 50

        # down_width = 128
        # down_points = (down_width, down_width)
        # image_resize = cv2.resize(image_gray, down_points, interpolation= cv2.INTER_LINEAR)
        # image_resize = image_resize / 255

        # image_reshaped = np.reshape(image_resize, (128, 128, -1)) TODO for 3D CNNs

        result.append(image)

    image_3D = np.stack(result, axis=0) # 2
    return image_3D


rootpath_CP = './data/dataset_seg/CP'
rootpath_NCP = './data/dataset_seg/NCP'
rootpath_Normal = './data/dataset_seg/Normal'

list_CP_train = []
list_CP_test = []

list_NCP_train = []
list_NCP_test = []

list_Normal_train = []
list_Normal_test = []

patients_CP_train = True
patients_NCP_train = True
patients_Normal_train = True
# list_CP = []
# list_NCP = []
# list_Normal = []

for idx, directory in enumerate(sorted(os.listdir(rootpath_CP))):
    subpath = os.path.join(rootpath_CP, directory)
    # check if train or test (len(train))
    if idx >= 771:
        patients_CP_train = False
    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)

        if len(os.listdir(subsubpath)) > 50:
            if patients_CP_train:
                image_3D = stack_2D_images(subsubpath, 'RANDOM')
                list_CP_train.append(image_3D)
            else:
                image_3D = stack_2D_images(subsubpath, 'SYMMETRIC')
                list_CP_test.append(image_3D)
print('CP done')

for idx, directory in enumerate(sorted(os.listdir(rootpath_NCP))):
    subpath = os.path.join(rootpath_NCP, directory)
    if idx >= 732:
        patients_NCP_train = False
    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)

        if len(os.listdir(subsubpath)) > 50:
            if patients_NCP_train:
                image_3D = stack_2D_images(subsubpath, 'RANDOM')
                list_NCP_train.append(image_3D)
            else:
                image_3D = stack_2D_images(subsubpath, 'SYMMETRIC')
                list_NCP_test.append(image_3D)

print('NCP done')

for idx, directory in enumerate(sorted(os.listdir(rootpath_Normal))):
    subpath = os.path.join(rootpath_Normal, directory)
    if idx >= 654:
        patients_Normal_train = False
    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)

        if len(os.listdir(subsubpath)) > 50:
            if patients_Normal_train:
                image_3D = stack_2D_images(subsubpath, 'RANDOM')
                list_Normal_train.append(image_3D)
            else:
                image_3D = stack_2D_images(subsubpath, 'SYMMETRIC')
                list_Normal_test.append(image_3D)
print('Normal done')

dataset_CP_train = np.stack(list_CP_train, axis = 0)
dataset_NCP_train = np.stack(list_NCP_train, axis = 0)
dataset_Normal_train = np.stack(list_Normal_train, axis = 0)

dataset_CP_test = np.stack(list_CP_test, axis = 0)
dataset_NCP_test = np.stack(list_NCP_test, axis = 0)
dataset_Normal_test = np.stack(list_Normal_test, axis = 0)

np.savez_compressed('data-arrays/dataset_CP_train_2', dataset_CP_train)
np.savez_compressed('data-arrays/dataset_NCP_train_2', dataset_NCP_train)
np.savez_compressed('data-arrays/dataset_Normal_train_2', dataset_Normal_train)

np.savez_compressed('data-arrays/dataset_CP_test_2', dataset_CP_test)
np.savez_compressed('data-arrays/dataset_NCP_test_2', dataset_NCP_test)
np.savez_compressed('data-arrays/dataset_Normal_test_2', dataset_Normal_test)

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