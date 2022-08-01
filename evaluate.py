import tensorflow as tf
from tensorflow import keras
from sklearn import metrics

import numpy as np
import pandas as pd

loader_CP_test = np.load('data-arrays-final/dataset_CP_test_small.npz')
loader_NCP_test = np.load('data-arrays-final/dataset_NCP_test_small.npz')
loader_Normal_test = np.load('data-arrays-final/dataset_Normal_test_small.npz')

dataset_CP_test = loader_CP_test['arr_0'] # 1176
dataset_NCP_test = loader_NCP_test['arr_0'] # 1280
dataset_Normal_test = loader_Normal_test['arr_0'] # 850

dataset_CP_test = dataset_CP_test.reshape(-1, 80, 64, 16)
dataset_NCP_test = dataset_NCP_test.reshape(-1, 80, 64, 16)
dataset_Normal_test = dataset_Normal_test.reshape(-1, 80, 64, 16)

dataset_CP_test = dataset_CP_test[:, :, :, :, np.newaxis]
dataset_NCP_test = dataset_NCP_test[:, :, :, :, np.newaxis]
dataset_Normal_test = dataset_Normal_test[:, :, :, :, np.newaxis]

# CP_labels = np.array([[1,0,0] for _ in range(len(dataset_CP_test))])
# NCP_labels = np.array([[0,1,0] for _ in range(len(dataset_NCP_test))])
# Normal_labels = np.array([[0,0,1] for _ in range(len(dataset_Normal_test))])

# x_test = np.concatenate((dataset_CP_test, dataset_NCP_test, dataset_Normal_test), axis=0)
# y_test = np.concatenate((CP_labels, NCP_labels, Normal_labels), axis=0)

CP_labels = np.array([[1,0] for _ in range(len(dataset_CP_test))])
NCP_labels = np.array([[0,1] for _ in range(len(dataset_NCP_test))])
Normal_labels = np.array([[1,0] for _ in range(len(dataset_Normal_test))])

x_test = np.concatenate((dataset_Normal_test, dataset_NCP_test), axis=0)
y_test = np.concatenate((Normal_labels, NCP_labels), axis=0)

print("data loaded")

model = keras.models.load_model("3d_densenet_2class_small_data.h5")

# results = model.evaluate(x_test, y_test, batch_size=32)
# print("test loss, test acc:", results)

predictions = model.predict(x_test)
print("predictions done")

# create y_pred: highest number goes to 1, rest to 0
for i in range(0, len(predictions)):
    x = np.where(predictions[i] == max(predictions[i]))
    predictions[i][x] = 1
    
    for j in range (0, 2):
        if predictions[i][j] < 1:
            predictions[i][j] = 0

predictions = predictions.astype(int)
print(metrics.confusion_matrix(y_test.argmax(axis = 1), predictions.argmax(axis = 1)))
print(metrics.classification_report(y_test.argmax(axis = 1), predictions.argmax(axis = 1), digits=3))

res = []
for l in [0,1]:
    prec,recall,_,_ = metrics.precision_recall_fscore_support(np.array(y_test.argmax(axis = 1))==l,
                                                      np.array(predictions.argmax(axis = 1))==l,
                                                      pos_label=True,average=None)
    res.append([l,recall[0],recall[1]])
pd.DataFrame(res,columns = ['class','sensitivity','specificity'])

# two_class_test = np.zeros((len(y_test), 2), dtype = int)
# two_class_pred = np.zeros((len(y_test), 2), dtype = int)

# for i in range(0, len(predictions)):
#     if y_test[i][0] == 1:
#         two_class_test[i][0] = 1
#     if y_test[i][1] == 1:
#         two_class_test[i][1] = 1
#     # if y_test[i][2] == 1:
#     #     two_class_test[i][1] = 1

#     if predictions[i][0] == 1:
#         two_class_pred[i][0] = 1
#     if predictions[i][1] == 1:
#         two_class_pred[i][1] = 1
#     # if predictions[i][2] == 1:
#     #     two_class_pred[i][1] = 1

# print(metrics.confusion_matrix(two_class_test.argmax(axis = 1), two_class_pred.argmax(axis = 1)))
# print(metrics.classification_report(two_class_test.argmax(axis = 1), two_class_pred.argmax(axis = 1), digits=2))
# # TP / (TP + FP)

# [[137  33  14]
#  [ 37  73  64]
#  [128  12  16]]

# TP FP
# FN TN
print("done")
