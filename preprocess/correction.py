import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from bounding_box import find_biggest_bounding_box_in_img


rootpath = '../data/dataset_seg'
labels = ['CP', 'NCP', 'Normal']

for label in labels:
    rootpath_label = rootpath + '/' + label

    for directory in sorted(os.listdir(rootpath_label)):
        subpath = os.path.join(rootpath_label, directory)

        
        for subdirectory in sorted(os.listdir(subpath)):
            subsubpath = os.path.join(subpath, subdirectory)
            first_time_zero = True
            zero_path = ''
            list_dir = sorted(os.listdir(subsubpath))

            for image_file in list_dir: #[::2]
                if image_file != '.ipynb_checkpoints':

                    img_path = os.path.join(subsubpath, image_file)
                    x, y, w, h = find_biggest_bounding_box_in_img(img_path)
                    if w > 0 and first_time_zero:
                        first_time_zero = False
                        # w = 100 # or if w < 0 and then increase for the second time -> 2 lungs
                        # or cut off everything when w < 0 for the second time
                    if w < 0 and not first_time_zero:
                        zero_path = img_path
                        print("break")
                        break

            #plot whole ct scan
            fig = plt.figure(figsize=(120, 120))
            
            columns = 10
            rows = 30
            i = 1
            for image_file in list_dir:
                if image_file != '.ipynb_checkpoints':
                    img_path = os.path.join(subsubpath, image_file)
                    im = cv.imread(img_path)
                    img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                    fig.add_subplot(rows, columns, i)
                    plt.imshow(img, cmap='gray')
                    i += 1
            plt.show()
            print(subsubpath)
            print(zero_path)
            
            
            

                

    print('label ' + label + ' done')