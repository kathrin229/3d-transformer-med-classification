import numpy as np
import matplotlib.pyplot as plt

loader_CP = np.load('../data-arrays-final/dataset_CP_train_2_scaled.npz')
loader_NCP = np.load('../data-arrays-final/dataset_NCP_train_2_scaled.npz')
loader_Normal = np.load('../data-arrays-final/dataset_NCP_train_2_scaled.npz')

dataset_CP = loader_CP['arr_0'] # 1176
dataset_NCP = loader_NCP['arr_0'] # 1280
dataset_Normal = loader_Normal['arr_0'] # 850

print('data loaded')

fig = plt.figure(figsize=(12, 12))
columns = 8
rows = 4
scan = dataset_Normal[100]
i = 1
for img in scan:
    img = img * 255
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    i += 1
    
plt.show()