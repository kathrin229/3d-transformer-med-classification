"""
Creating two smaller datasets from preprocessed CC-CCII dataset size 160x128x64:
The image size and the number of samples are reduced by half.
The dataset 80x64x16-mid takes the 16 slices in the middle of each scan for downsampling.
The dataset 80x64x16-mid takes every second slice of each scan for downsampling.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

loader_CP_train = np.load('Data/dataset_CP_train_160x128x32.npz')
loader_NCP_train = np.load('Data/dataset_NCP_train_160x128x32.npz')
loader_Normal_train = np.load('Data/dataset_Normal_train_160x128x32.npz')

loader_CP_valid = np.load('Data/dataset_CP_valid_160x128x32.npz')
loader_NCP_valid = np.load('Data/dataset_NCP_valid_160x128x32.npz')
loader_Normal_valid = np.load('Data/dataset_Normal_valid_160x128x32.npz')

loader_CP_test = np.load('Data/dataset_CP_test_160x128x32.npz')
loader_NCP_test = np.load('Data/dataset_NCP_test_160x128x32.npz')
loader_Normal_test = np.load('Data/dataset_Normal_test_160x128x32.npz')

dataset_CP_train = loader_CP_train['arr_0']
dataset_NCP_train = loader_NCP_train['arr_0']
dataset_Normal_train = loader_Normal_train['arr_0']

dataset_CP_valid = loader_CP_valid['arr_0']
dataset_NCP_valid = loader_NCP_valid['arr_0']
dataset_Normal_valid = loader_Normal_valid['arr_0']

dataset_CP_test = loader_CP_test['arr_0']
dataset_NCP_test = loader_NCP_test['arr_0']
dataset_Normal_test = loader_Normal_test['arr_0']

print(len(dataset_CP_test))
print(len(dataset_NCP_test))
print(print(len(dataset_Normal_test)))

SAMPLING = '2ND'  # 'MID'

datasets = [dataset_CP_train, dataset_NCP_train, dataset_Normal_train,
            dataset_CP_valid, dataset_NCP_valid, dataset_Normal_valid,
            dataset_CP_test, dataset_NCP_test, dataset_Normal_test]

dataset_names = ['dataset_CP_train', 'dataset_NCP_train', 'dataset_Normal_train',
            'dataset_CP_valid', 'dataset_NCP_valid', 'dataset_Normal_valid',
            'dataset_CP_test', 'dataset_NCP_test', 'dataset_Normal_test']

for dataset, dataset_name in zip(datasets, dataset_names):
    list_dataset = []
    for scan in dataset:
        new_scan = []
        for idx, img in enumerate(scan):
            img = img * 255
            img_resize = cv.resize(img, (80, 64), interpolation=cv.INTER_AREA)  # resize image .INTER_LINEAR
            new_scan.append(img_resize)
        if SAMPLING == '2ND':
            sample_new = new_scan[::2]
        elif SAMPLING == 'MID':
            sample_new = new_scan[8:24]
        list_dataset.append(sample_new)
    dataset_new = np.stack(list_dataset, axis=0)
    print(dataset_name)

    if SAMPLING == '2ND':
        np.savez_compressed('Data/%s_80x64x16-2nd.npz' % dataset_name, dataset_new)
    elif SAMPLING == 'MID':
        np.savez_compressed('Data/%s_80x64x16-mid.npz' % dataset_name, dataset_new)

print('done')