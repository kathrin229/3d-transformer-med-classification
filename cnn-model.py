import numpy as np

# https://keras.io/examples/vision/3D_image_classification/#loading-data-and-preprocessing

dataset_CP = np.load('data-arrays/dataset_CP.npy') # 1176
dataset_NCP = np.load('data-arrays/dataset_NCP.npy') # 1280
dataset_Normal = np.load('data-arrays/dataset_Normal.npy') # 850

CP_labels = np.array([[1,0,0] for _ in range(len(dataset_CP))])
NCP_labels = np.array([[0,1,0] for _ in range(len(dataset_NCP))])
Normal_labels = np.array([[0,0,1] for _ in range(len(dataset_Normal))])

# x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
# y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
# x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
# y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)

# train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

# batch_size = 2
# # Augment the on the fly during training.
# train_dataset = (
#     train_loader.shuffle(len(x_train))
#     .map(train_preprocessing)
#     .batch(batch_size)
#     .prefetch(2)
# )
# # Only rescale.
# validation_dataset = (
#     validation_loader.shuffle(len(x_val))
#     .map(validation_preprocessing)
#     .batch(batch_size)
#     .prefetch(2)
# )

# print("Number of samples in train and validation are %d and %d."
#     % (x_train.shape[0], x_val.shape[0]))

