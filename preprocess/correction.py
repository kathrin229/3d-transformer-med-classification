import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from bounding_box import find_biggest_bounding_box_in_img
from data_preprocessing_functions import find_bounding_box_size


loader_CP = np.load('./data-arrays/dataset_CP_train_5_scaled.npz')
loader_NCP = np.load('./data-arrays/dataset_NCP_train_5_scaled.npz')
loader_Normal = np.load('./data-arrays/dataset_Normal_train_5_scaled.npz')

loader_CP_valid = np.load('./data-arrays/dataset_CP_valid_5_scaled.npz')
loader_NCP_valid = np.load('./data-arrays/dataset_NCP_valid_5_scaled.npz')
loader_Normal_valid = np.load('./data-arrays/dataset_Normal_valid_5_scaled.npz')

loader_CP_test = np.load('./data-arrays/dataset_CP_test_5_scaled.npz')
loader_NCP_test = np.load('./data-arrays/dataset_NCP_test_5_scaled.npz')
loader_Normal_test = np.load('./data-arrays/dataset_Normal_test_5_scaled.npz')

dataset_CP = loader_CP['arr_0'] # 1176
dataset_NCP = loader_NCP['arr_0'] # 1280
dataset_Normal = loader_Normal['arr_0'] # 850

dataset_CP_valid = loader_CP_valid['arr_0'] # 1176
dataset_NCP_valid = loader_NCP_valid['arr_0'] # 1280
dataset_Normal_valid = loader_Normal_valid['arr_0'] # 850

dataset_CP_test = loader_CP_test['arr_0'] # 1176
dataset_NCP_test = loader_NCP_test['arr_0'] # 1280
dataset_Normal_test = loader_Normal_test['arr_0'] # 850

print("data loaded")

print(len(dataset_CP))
print(len(dataset_NCP))
print(len(dataset_Normal))
print("")

print(len(dataset_CP_valid))
print(len(dataset_NCP_valid))
print(len(dataset_Normal_valid))
print("")

print(len(dataset_CP_test))
print(len(dataset_NCP_test))
print(len(dataset_Normal_test))

# d_CP = 0
# d_NCP = 0
# d_Normal = 0

# d_CP_valid = 0
# d_NCP_valid = 0
# d_Normal_valid = 0

# d_CP_test = 0
# d_NCP_test = 0
# d_Normal_test = 0

discard_list = []

dataset_list = [dataset_CP, dataset_NCP, dataset_Normal,
                dataset_CP_valid, dataset_NCP_valid, dataset_Normal_valid,
                dataset_CP_test, dataset_NCP_test, dataset_Normal_test]

for dataset_idx, dataset in enumerate(dataset_list):
    num_discard = 0
    scan_d_list = []
    for scan_idx, scan in enumerate(dataset):
        discard = False
        w_check = 0
        w_first = 0
        h_first = 0
        for idx, img in enumerate(scan):

            x, y, w, h = find_bounding_box_size(img, 118, 160)

            if idx == 0:
                w_first = w
                h_first = h

            if idx == len(scan)-1 and w < w_first and h < h_first and discard == False:
                discard = True
                # print(str(number) + ": discard")
                num_discard +=1

            if idx <= 5 and w < w_check -2 and discard == False:
                discard = True
                # print(str(number) + ": discard")
                num_discard +=1
            w_check = w

            if idx > 8 and idx <= 24 and w < 90 and discard == False:
                discard = True
                # print(str(number) + ": discard")
                num_discard +=1
        if discard:
            scan_d_list.append(scan_idx)
    dataset = np.delete(dataset, scan_d_list, axis=0)
    dataset_list[dataset_idx] = dataset
    discard_list.append(num_discard)
print(discard_list)

np.savez_compressed('data-arrays/dataset_CP_train_5_corrected', dataset_list[0])
np.savez_compressed('data-arrays/dataset_NCP_train_5_corrected', dataset_list[1])
np.savez_compressed('data-arrays/dataset_Normal_train_5_corrected', dataset_list[2])

np.savez_compressed('data-arrays/dataset_CP_valid_5_corrected', dataset_list[3])
np.savez_compressed('data-arrays/dataset_NCP_valid_5_corrected', dataset_list[4])
np.savez_compressed('data-arrays/dataset_Normal_valid_5_corrected', dataset_list[5])

np.savez_compressed('data-arrays/dataset_CP_test_5_corrected', dataset_list[6])
np.savez_compressed('data-arrays/dataset_NCP_test_5_corrected', dataset_list[7])
np.savez_compressed('data-arrays/dataset_Normal_test_5_corrected', dataset_list[8])
#50 - 25 - 25
#60 - 15 - 25
#60 - 15 - 25
print(len(dataset_list[0]))
print(len(dataset_list[1]))
print(len(dataset_list[2]))
print("")

print(len(dataset_list[3]))
print(len(dataset_list[4]))
print(len(dataset_list[5]))
print("")

print(len(dataset_list[6]))
print(len(dataset_list[7]))
print(len(dataset_list[8]))

# print(len(dataset_CP) - discard_list[0])
# print(len(dataset_NCP)- discard_list[1])
# print(len(dataset_Normal)- discard_list[2])
# print("")

# print(len(dataset_CP_valid)- discard_list[3])
# print(len(dataset_NCP_valid)- discard_list[4])
# print(len(dataset_Normal_valid)- discard_list[5])
# print("")

# print(len(dataset_CP_test)- discard_list[6])
# print(len(dataset_NCP_test)- discard_list[7])
# print(len(dataset_Normal_test)- discard_list[8])

print('done')
'''
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
'''