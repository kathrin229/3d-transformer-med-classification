"""
Visualizing lung scans from the created dataset. (Run current file in interactive window)
"""

import numpy as np
import matplotlib.pyplot as plt
from bounding_box import find_bounding_box_size

loader= np.load('../Data/dataset_CP_test_160x128x32_new.npz')
dataset = loader['arr_0']
print('data loaded')

numbers = [42]  # [i for i in range(1, 117)]

for number in numbers:
    fig = plt.figure(figsize=(12, 12))
    columns = 8
    rows = 4
    scan = dataset[number]
    i = 1
    for img in scan:
        img = img * 255
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
        i += 1

plt.show()

print('done')