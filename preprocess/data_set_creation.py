"""
Creating an input dataset from the Clean CC-CCII Dataset https://github.com/wang-shihao/HKBU_HPML_COVID-19
Training, validation and test set are saved as .npz files separately for every class
"""

import os
import cv2
import random
import numpy as np

from data_preprocessing import stack_2D_images
from data_preprocessing import correct_datasets

rootpath_CP = 'dataset_seg/CP'
rootpath_NCP = 'dataset_seg/NCP'
rootpath_Normal = 'dataset_seg/Normal'

NUM_SLICES_PER_SCAN = 32
WINDOW_W = 480
WINDOW_H = 384
FINAL_W = 160
FINAL_H = 128

PERCENT_TRAIN = 0.60
PERCENT_VALID = 0.15
PERCENT_TEST = 0.25

PERCENT_TRAIN_CP = 0.40
PERCENT_VALID_CP = 0.15
PERCENT_TEST_CP = 0.45

list_CP_train = []
list_CP_valid = []
list_CP_test = []

list_NCP_train = []
list_NCP_valid = []
list_NCP_test = []

list_Normal_train = []
list_Normal_valid = []
list_Normal_test = []

num_patients_CP = 964
num_patients_NCP = 916
num_patients_Normal = 818

patients_CP_train = True
patients_NCP_train = True
patients_Normal_train = True

patients_CP_valid= False
patients_NCP_valid= False
patients_Normal_valid = False


for idx, directory in enumerate(sorted(os.listdir(rootpath_CP))):
    subpath = os.path.join(rootpath_CP, directory)
    if idx >= int(num_patients_CP * PERCENT_TRAIN_CP) and idx < int(num_patients_CP * PERCENT_TRAIN_CP) + int(num_patients_CP * PERCENT_VALID_CP):
        patients_CP_train = False
        patients_CP_valid = True
    if idx >= int(num_patients_CP * PERCENT_TRAIN_CP) + int(num_patients_CP * PERCENT_VALID_CP):
        patients_CP_train = False
        patients_CP_valid = False

    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)
        list_dir = sorted(os.listdir(subsubpath))
        if len(list_dir) > NUM_SLICES_PER_SCAN:
            if patients_CP_train:
                image_3D = stack_2D_images(subsubpath, list_dir, 'SYMMETRIC', NUM_SLICES_PER_SCAN, WINDOW_W, WINDOW_H, FINAL_W, FINAL_H)
                list_CP_train.append(image_3D)
            elif patients_CP_valid:
                image_3D = stack_2D_images(subsubpath, list_dir, 'SYMMETRIC', NUM_SLICES_PER_SCAN, WINDOW_W, WINDOW_H, FINAL_W, FINAL_H)
                list_CP_valid.append(image_3D)
            else:
                image_3D = stack_2D_images(subsubpath, list_dir, 'SYMMETRIC', NUM_SLICES_PER_SCAN, WINDOW_W, WINDOW_H, FINAL_W, FINAL_H)
                list_CP_test.append(image_3D)
print('CP done')

for idx, directory in enumerate(sorted(os.listdir(rootpath_NCP))):
    subpath = os.path.join(rootpath_NCP, directory)
    if idx >= int(num_patients_NCP * PERCENT_TRAIN) and idx < int(num_patients_NCP * PERCENT_TRAIN) + int(num_patients_NCP * PERCENT_VALID):
        patients_NCP_train = False
        patients_NCP_valid = True
    if idx >= int(num_patients_NCP * PERCENT_TRAIN) + int(num_patients_NCP * PERCENT_VALID):
        patients_NCP_train = False
        patients_NCP_valid = False

    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)
        list_dir = sorted(os.listdir(subsubpath))
        if len(list_dir) > NUM_SLICES_PER_SCAN:
            if patients_NCP_train:
                image_3D = stack_2D_images(subsubpath, list_dir, 'SYMMETRIC', NUM_SLICES_PER_SCAN, WINDOW_W, WINDOW_H, FINAL_W, FINAL_H)
                list_NCP_train.append(image_3D)
            elif patients_NCP_valid:
                image_3D = stack_2D_images(subsubpath, list_dir, 'SYMMETRIC', NUM_SLICES_PER_SCAN, WINDOW_W, WINDOW_H, FINAL_W, FINAL_H)
                list_NCP_valid.append(image_3D)
            else:
                image_3D = stack_2D_images(subsubpath, list_dir, 'SYMMETRIC', NUM_SLICES_PER_SCAN, WINDOW_W, WINDOW_H, FINAL_W, FINAL_H)
                list_NCP_test.append(image_3D)

print('NCP done')

for idx, directory in enumerate(sorted(os.listdir(rootpath_Normal))):
    subpath = os.path.join(rootpath_Normal, directory)
    if idx >= int(num_patients_Normal * PERCENT_TRAIN) and idx < int(num_patients_Normal * PERCENT_TRAIN) + int(num_patients_Normal * PERCENT_VALID):
        patients_Normal_train = False
        patients_Normal_valid = True
    if idx >= int(num_patients_Normal * PERCENT_TRAIN) + int(num_patients_Normal * PERCENT_VALID):
        patients_Normal_train = False
        patients_Normal_valid = False

    for subdirectory in sorted(os.listdir(subpath)):
        subsubpath = os.path.join(subpath, subdirectory)
        list_dir = sorted(os.listdir(subsubpath))
        if len(list_dir) > NUM_SLICES_PER_SCAN:
            if patients_Normal_train:
                image_3D = stack_2D_images(subsubpath, list_dir, 'SYMMETRIC', NUM_SLICES_PER_SCAN, WINDOW_W, WINDOW_H, FINAL_W, FINAL_H)
                list_Normal_train.append(image_3D)
            elif patients_Normal_valid:
                image_3D = stack_2D_images(subsubpath, list_dir, 'SYMMETRIC', NUM_SLICES_PER_SCAN, WINDOW_W, WINDOW_H, FINAL_W, FINAL_H)
                list_Normal_valid.append(image_3D)
            else:
                image_3D = stack_2D_images(subsubpath, list_dir, 'SYMMETRIC', NUM_SLICES_PER_SCAN, WINDOW_W, WINDOW_H, FINAL_W, FINAL_H)
                list_Normal_test.append(image_3D)
print('Normal done')


dataset_CP_train = np.stack(list_CP_train, axis = 0)
dataset_NCP_train = np.stack(list_NCP_train, axis = 0)
dataset_Normal_train = np.stack(list_Normal_train, axis = 0)

dataset_CP_valid = np.stack(list_CP_valid, axis = 0)
dataset_NCP_valid = np.stack(list_NCP_valid, axis = 0)
dataset_Normal_valid = np.stack(list_Normal_valid, axis = 0)

dataset_CP_test = np.stack(list_CP_test, axis = 0)
dataset_NCP_test = np.stack(list_NCP_test, axis = 0)
dataset_Normal_test = np.stack(list_Normal_test, axis = 0)

datasets = [dataset_CP_train, dataset_NCP_train, dataset_Normal_train,
            dataset_CP_valid, dataset_NCP_valid, dataset_Normal_valid,
            dataset_CP_test, dataset_NCP_test, dataset_Normal_test]

dataset_list, discard_list = correct_datasets(datasets, FINAL_W, FINAL_H)
print(sum(discard_list))

np.savez_compressed('Data/dataset_CP_train_160x128x32', dataset_list[0])
np.savez_compressed('Data/dataset_NCP_train_160x128x32', dataset_list[1])
np.savez_compressed('Data/dataset_Normal_train_160x128x32', dataset_list[2])

np.savez_compressed('Data/dataset_CP_valid_160x128x32', dataset_list[3])
np.savez_compressed('Data/dataset_NCP_valid_160x128x32', dataset_list[4])
np.savez_compressed('Data/dataset_Normal_valid_160x128x32', dataset_list[5])

np.savez_compressed('Data/dataset_CP_test_160x128x32', dataset_list[6])
np.savez_compressed('Data/dataset_NCP_test_160x128x32', dataset_list[7])
np.savez_compressed('Data/dataset_Normal_test_160x128x32', dataset_list[8])

print('done')