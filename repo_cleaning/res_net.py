import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from DenseNet3D import DenseNet3D

# TODO load also test set
loader_CP = np.load('data-arrays-final/dataset_CP_train_3_scaled.npz')
loader_NCP = np.load('data-arrays-final/dataset_NCP_train_3_scaled.npz')
loader_Normal = np.load('data-arrays-final/dataset_Normal_train_3_scaled.npz')

loader_CP_test = np.load('data-arrays-final/dataset_CP_test_3_scaled.npz')
loader_NCP_test = np.load('data-arrays-final/dataset_NCP_test_3_scaled.npz')
loader_Normal_test = np.load('data-arrays-final/dataset_Normal_test_3_scaled.npz')

dataset_CP = loader_CP['arr_0'] # 1176
dataset_NCP = loader_NCP['arr_0'] # 1280
dataset_Normal = loader_Normal['arr_0'] # 850

dataset_CP_test = loader_CP_test['arr_0'] # 1176
dataset_NCP_test = loader_NCP_test['arr_0'] # 1280
dataset_Normal_test = loader_Normal_test['arr_0'] # 850

print('data loaded')

dataset_CP = dataset_CP.reshape(-1, 118, 160, 32)
dataset_NCP = dataset_NCP.reshape(-1, 118, 160, 32)
dataset_Normal = dataset_Normal.reshape(-1, 118, 160, 32)

dataset_CP = dataset_CP[:, :, :, :, np.newaxis]
dataset_NCP = dataset_NCP[:, :, :, :, np.newaxis]
dataset_Normal = dataset_Normal[:, :, :, :, np.newaxis]

# CP_labels = np.array([[1,0,0] for _ in range(len(dataset_CP))])
# NCP_labels = np.array([[0,1,0] for _ in range(len(dataset_NCP))])
# Normal_labels = np.array([[0,0,1] for _ in range(len(dataset_Normal))])

# x_train = np.concatenate((dataset_CP [:int(len(dataset_CP)*0.8)], dataset_NCP[:int(len(dataset_NCP)*0.8)], dataset_Normal[:int(len(dataset_Normal)*0.8)]), axis=0)
# y_train = np.concatenate((CP_labels[:int(len(dataset_CP)*0.8)], NCP_labels[:int(len(dataset_NCP)*0.8)], Normal_labels[:int(len(dataset_Normal)*0.8)]), axis=0)
# x_val = np.concatenate((dataset_CP[int(len(dataset_CP)*0.8):], dataset_NCP[int(len(dataset_NCP)*0.8):], dataset_Normal[int(len(dataset_Normal)*0.8):]), axis=0)
# y_val = np.concatenate((CP_labels[int(len(dataset_CP)*0.8):], NCP_labels[int(len(dataset_NCP)*0.8):], Normal_labels[int(len(dataset_Normal)*0.8):]), axis=0)

CP_labels = np.array([[1,0] for _ in range(len(dataset_CP))])
NCP_labels = np.array([[0,1] for _ in range(len(dataset_NCP))])

x_train = np.concatenate((dataset_CP [:int(len(dataset_CP)*0.8)], dataset_NCP[:int(len(dataset_NCP)*0.8)]), axis=0)
y_train = np.concatenate((CP_labels[:int(len(dataset_CP)*0.8)], NCP_labels[:int(len(dataset_NCP)*0.8)]), axis=0)
x_val = np.concatenate((dataset_CP[int(len(dataset_CP)*0.8):], dataset_NCP[int(len(dataset_NCP)*0.8):]), axis=0)
y_val = np.concatenate((CP_labels[int(len(dataset_CP)*0.8):], NCP_labels[int(len(dataset_NCP)*0.8):]), axis=0)

train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

print('input created')

batch_size = 16
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .batch(batch_size)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .batch(batch_size)
)

print("Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0]))


# DenseNet3D(input_shape=None,
#                depth=40,
#                nb_dense_block=3,
#                growth_rate=12,
#                nb_filter=-1,
#                nb_layers_per_block=-1,
#                bottleneck=False,
#                reduction=0.0,
#                dropout_rate=0.0,
#                weight_decay=1e-4,
#                subsample_initial_block=False,
#                include_top=True,
#                input_tensor=None,
#                pooling=None,
#                classes=10,
#                activation='softmax',
#                transition_pooling='avg'):
# model = DenseNet3D(input_shape = (118, 160, 32, 1),
#                     classes = 3)





# https://github.com/JihongJu/keras-resnet3d#readme
from resnet3d import Resnet3DBuilder

model = Resnet3DBuilder.build_resnet_50((118, 160, 32, 1), 2) #3

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification_2_class_pneumonia.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15)

# model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(  x=x_train,
            y=y_train,
            batch_size=32,
            epochs=15,
            callbacks=[checkpoint_cb, early_stopping_cb],
            validation_data=validation_dataset)

# def fit(self,
#           x=None,
#           y=None,
#           batch_size=None,
#           epochs=1,
#           verbose='auto',
#           callbacks=None,
#           validation_split=0.,
#           validation_data=None,
#           shuffle=True,
#           class_weight=None,
#           sample_weight=None,
#           initial_epoch=0,
#           steps_per_epoch=None,
#           validation_steps=None,
#           validation_batch_size=None,
#           validation_freq=1,
#           max_queue_size=10,
#           workers=1,
#           use_multiprocessing=False):


# initial_learning_rate = 0.0001
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
# )
# model.compile(
#     loss="binary_crossentropy",
#     optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
#     metrics=["acc"],
# )


fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["accuracy", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
plt.savefig('resnet_two_class_pneumonia_new_dataset.png')

print("done")