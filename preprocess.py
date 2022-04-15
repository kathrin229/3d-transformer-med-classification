
import os
# import cv2
import cv2
print(cv2.__version__)

# import matplotlib

import matplotlib.pyplot as plt
import matplotlib.image as img


# TODO Count images per person
# TODO Sample scans with 50 files
# TODO Resize images
# TODO add to matrix

rootdir = '../../data/dataset_seg/'

# for file in os.listdir(rootdir):
#     d = os.path.join(rootdir, file)
#     print(d)

path = '../../data/dataset_seg/CP/1065/3104/0022.png'

image = cv2.imread(path)

# cv2.imshow("Display window", image)

print(type(image))
# <class 'numpy.ndarray'>

print(image.shape)
print(type(image.shape))

# image = img.imread(path)
# plt.imshow(image)
# plt.show()


print("done")