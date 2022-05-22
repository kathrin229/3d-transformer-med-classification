import numpy as np
import matplotlib.pyplot as plt

loader_CP = np.load('../data-arrays-final/dataset_CP_train_2_scaled.npz')
loader_NCP = np.load('../data-arrays-final/dataset_NCP_train_2_scaled.npz')
loader_Normal = np.load('../data-arrays-final/dataset_NCP_train_2_scaled.npz')

loader_CP_test = np.load('../data-arrays-final/dataset_CP_test_2_scaled.npz')
loader_NCP_test = np.load('../data-arrays-final/dataset_NCP_test_2_scaled.npz')
loader_Normal_test = np.load('../data-arrays-final/dataset_Normal_test_2_scaled.npz')

dataset_CP = loader_CP['arr_0'] # 1176
dataset_NCP = loader_NCP['arr_0'] # 1280
dataset_Normal = loader_Normal['arr_0'] # 850

dataset_CP_test = loader_CP_test['arr_0'] # 1176
dataset_NCP_test = loader_NCP_test['arr_0'] # 1280
dataset_Normal_test = loader_Normal_test['arr_0'] # 850

print('data loaded')

#numbers = [2, 20, 37, 100, 374, 659, 701, 806]
numbers = [21,22]

for number in numbers:
    fig = plt.figure(figsize=(12, 12))
    columns = 8
    rows = 4
    scan = dataset_Normal_test[number]
    i = 1
    for img in scan:
        img = img * 255
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
        i += 1

for number in numbers:
    fig = plt.figure(figsize=(12, 12))
    columns = 8
    rows = 4
    scan = dataset_Normal[number]
    i = 1
    for img in scan:
        img = img * 255
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
        i += 1
    
plt.show()