import numpy as np
import matplotlib.pyplot as plt
from bounding_box import find_bounding_box_size

# loader_CP = np.load('../data-arrays/dataset_CP_train_4_scaled.npz')
loader_NCP = np.load('../data-arrays/dataset_CP_test_5_corrected.npz')
# loader_NCP = np.load('../data-arrays/dataset_NCP_train_99_discard.npz')
# loader_Normal = np.load('../data-arrays/dataset_Normal_train_4_scaled.npz')

# loader_CP_test = np.load('../data-arrays/dataset_CP_test_4_scaled.npz')
# loader_NCP_test = np.load('../data-arrays/dataset_NCP_test_4_scaled.npz')
# loader_Normal_test = np.load('../data-arrays/dataset_Normal_test_4_scaled.npz')

# dataset_CP = loader_CP['arr_0'] # 1176
dataset_NCP = loader_NCP['arr_0'] # 1280
# dataset_Normal = loader_Normal['arr_0'] # 850

# dataset_CP_test = loader_CP_test['arr_0'] # 1176
# dataset_NCP_test = loader_NCP_test['arr_0'] # 1280
# dataset_Normal_test = loader_Normal_test['arr_0'] # 850

print('data loaded')



# numbers = [2, 20, 37, 100, 374, 659, 701, 806]
#numbers = [21,22]
numbers = [i for i in range(1, 117)]
# numbers = [94, 95, 96, 97, 98, 99]
# numbers = [3, 4, 6, 7, 98, 69, 56, 55, 32, 24]#, 90, 91, 92, 93, 94, 95]
# numbers = [190, 177, 176, 172, 159, 158, 155, 154, 153, 152]
# fig = plt.figure(figsize=(12, 120))
# columns = 2
# rows = 100
# i = 1
# for scan in dataset_NCP[1100:1200]:

#     fig.add_subplot(rows, columns, i)
#     plt.imshow(scan[16]*255, cmap='gray')
#     i += 1

# for number in numbers:
#     fig = plt.figure(figsize=(12, 12))
#     columns = 8
#     rows = 4
#     scan = dataset_Normal_test[number]
#     i = 1
#     for img in scan:
#         img = img * 255
#         fig.add_subplot(rows, columns, i)
#         plt.imshow(img, cmap='gray')
#         i += 1

# Discard cases:
# new lung starts in the middle (lung w<90)
# random big and small slices in the beginning (decrease in first 5) Threshold? check 98
# reversed lung (w last slice < w first slice)
#98, 69, 56, 55, 32
# 24


# change random big and small slices in the beginning: threshold? See 98 and 94
# reversed slice: also criterion h last slice < h first slice?
num_discard = 0

for number in numbers:
    fig = plt.figure(figsize=(20, 12))
    columns = 32
    rows = 1
    scan = dataset_NCP[number]
    # scan = np.flipud(scan)
    # print(number)
    i = 1
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
        # print("width " + str(w) + " height " + str(h))
        img = img * 255

        fig.add_subplot(rows, columns, i)
        if discard:
            plt.title(str(number) + "d")# + ":w" + str(w) + " h" + str(h) )
        else:
            plt.title(str(number))# + ":w" + str(w) + " h" + str(h) )
        plt.imshow(img, cmap='gray')

        i += 1
    
    
plt.show()

print("Total discard: " +str(num_discard))