import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

from dense_net_2D import densenet
from dense_net_3D import densenet3D
# import medmnist


loader_CP = np.load('./data-arrays/dataset_CP_train_5_corrected.npz')
loader_NCP = np.load('./data-arrays/dataset_NCP_train_5_corrected.npz')
loader_Normal = np.load('./data-arrays/dataset_Normal_train_5_corrected.npz')

loader_CP_valid = np.load('./data-arrays/dataset_CP_valid_5_corrected.npz')
loader_NCP_valid = np.load('./data-arrays/dataset_NCP_valid_5_corrected.npz')
loader_Normal_valid = np.load('./data-arrays/dataset_Normal_valid_5_corrected.npz')

loader_CP_test = np.load('./data-arrays/dataset_CP_test_5_corrected.npz')
loader_NCP_test = np.load('./data-arrays/dataset_NCP_test_5_corrected.npz')
loader_Normal_test = np.load('./data-arrays/dataset_Normal_test_5_corrected.npz')

dataset_CP = loader_CP['arr_0'] # 1176
dataset_NCP = loader_NCP['arr_0'] # 1280
dataset_Normal = loader_Normal['arr_0'] # 850

dataset_CP_valid = loader_CP_valid['arr_0'] # 1176
dataset_NCP_valid = loader_NCP_valid['arr_0'] # 1280
dataset_Normal_valid = loader_Normal_valid['arr_0'] # 850

dataset_CP_test = loader_CP_test['arr_0'] # 1176
dataset_NCP_test = loader_NCP_test['arr_0'] # 1280
dataset_Normal_test = loader_Normal_test['arr_0'] # 850

dataset_CP = dataset_CP.reshape(-1, 160, 128, 32)
dataset_NCP = dataset_NCP.reshape(-1, 160, 128, 32)
dataset_Normal = dataset_Normal.reshape(-1, 160, 128, 32)

dataset_CP_valid = dataset_CP_valid.reshape(-1, 160, 128, 32)
dataset_NCP_valid = dataset_NCP_valid.reshape(-1, 160, 128, 32)
dataset_Normal_valid = dataset_Normal_valid.reshape(-1, 160, 128, 32)

dataset_CP_test = dataset_CP_test.reshape(-1, 160, 128, 32)
dataset_NCP_test = dataset_NCP_test.reshape(-1, 160, 128, 32)
dataset_Normal_test = dataset_Normal_test.reshape(-1, 160, 128, 32)

dataset_CP = dataset_CP[:, :, :, :, np.newaxis]
dataset_NCP = dataset_NCP[:, :, :, :, np.newaxis]
dataset_Normal = dataset_Normal[:, :, :, :, np.newaxis]
dataset_CP_valid = dataset_CP_valid[:, :, :, :, np.newaxis]
dataset_NCP_valid = dataset_NCP_valid[:, :, :, :, np.newaxis]
dataset_Normal_valid = dataset_Normal_valid[:, :, :, :, np.newaxis]
dataset_CP_test = dataset_CP_test[:, :, :, :, np.newaxis]
dataset_NCP_test = dataset_NCP_test[:, :, :, :, np.newaxis]
dataset_Normal_test = dataset_Normal_test[:, :, :, :, np.newaxis]

#### 3 class
# CP_labels = np.array([[1,0,0] for _ in range(len(dataset_CP))])
# NCP_labels = np.array([[0,1,0] for _ in range(len(dataset_NCP))])
# Normal_labels = np.array([[0,0,1] for _ in range(len(dataset_Normal))])

#### 2 class
CP_labels = np.array([[1,0] for _ in range(len(dataset_CP))])
# Normal_labels = np.array([[1,0] for _ in range(len(dataset_Normal))])
NCP_labels = np.array([[0,1] for _ in range(len(dataset_NCP))])

#### 3 class
# x_train = np.concatenate((dataset_CP, dataset_NCP, dataset_Normal), axis=0)
# y_train = np.concatenate((CP_labels, NCP_labels, Normal_labels), axis=0)

#### 2 class
x_train = np.concatenate((dataset_CP, dataset_NCP), axis=0)
y_train = np.concatenate((CP_labels, NCP_labels), axis=0)
# x_train = np.concatenate((dataset_Normal, dataset_NCP), axis=0)
# y_train = np.concatenate((Normal_labels, NCP_labels), axis=0)

#### 3 class
# CP_labels_valid = np.array([[1,0,0] for _ in range(len(dataset_CP_valid))])
# NCP_labels_valid = np.array([[0,1,0] for _ in range(len(dataset_NCP_valid))])
# Normal_labels_valid = np.array([[0,0,1] for _ in range(len(dataset_Normal_valid))])

#### 2 class
CP_labels_valid = np.array([[1,0] for _ in range(len(dataset_CP_valid))])
# Normal_labels_valid = np.array([[1,0] for _ in range(len(dataset_Normal_valid))])
NCP_labels_valid = np.array([[0,1] for _ in range(len(dataset_NCP_valid))])

#### 3 class
# x_val = np.concatenate((dataset_CP_valid, dataset_NCP_valid, dataset_Normal_valid), axis=0)
# y_val = np.concatenate((CP_labels_valid, NCP_labels_valid, Normal_labels_valid), axis=0)

#### 2 class
x_val = np.concatenate((dataset_CP_valid, dataset_NCP_valid), axis=0)
y_val = np.concatenate((CP_labels_valid, NCP_labels_valid), axis=0)
# x_val = np.concatenate((dataset_Normal_valid, dataset_NCP_valid), axis=0)
# y_val = np.concatenate((Normal_labels_valid, NCP_labels_valid), axis=0)

# x_train = np.concatenate((dataset_CP [:int(len(dataset_CP)*0.7)], dataset_NCP[:int(len(dataset_NCP)*0.7)], dataset_Normal[:int(len(dataset_Normal)*0.7)]), axis=0)
# y_train = np.concatenate((CP_labels[:int(len(dataset_CP)*0.7)], NCP_labels[:int(len(dataset_NCP)*0.7)], Normal_labels[:int(len(dataset_Normal)*0.7)]), axis=0)
# x_val = np.concatenate((dataset_CP[int(len(dataset_CP)*0.7):int(len(dataset_CP)*0.85)], dataset_NCP[int(len(dataset_NCP)*0.7):int(len(dataset_CP)*0.85)], dataset_Normal[int(len(dataset_Normal)*0.7):int(len(dataset_CP)*0.85)]), axis=0)
# y_val = np.concatenate((CP_labels[int(len(dataset_CP)*0.7):int(len(dataset_CP)*0.85)], NCP_labels[int(len(dataset_NCP)*0.7):int(len(dataset_CP)*0.85)], Normal_labels[int(len(dataset_Normal)*0.7):int(len(dataset_CP)*0.85)]), axis=0)

#### 3 class
# CP_labels_test = np.array([[1,0,0] for _ in range(len(dataset_CP_test))])
# NCP_labels_test = np.array([[0,1,0] for _ in range(len(dataset_NCP_test))])
# Normal_labels_test = np.array([[0,0,1] for _ in range(len(dataset_Normal_test))])

#### 2 class
CP_labels_test = np.array([[1,0] for _ in range(len(dataset_CP_test))])
# Normal_labels_test = np.array([[1,0] for _ in range(len(dataset_Normal_test))])
NCP_labels_test = np.array([[0,1] for _ in range(len(dataset_NCP_test))])

#### 3 class
# x_test = np.concatenate((dataset_CP_test, dataset_NCP_test, dataset_Normal_test), axis=0)
# y_test = np.concatenate((CP_labels_test, NCP_labels_test, Normal_labels_test), axis=0)


# x_test = np.concatenate((dataset_CP[int(len(dataset_CP)*0.85):], dataset_NCP[int(len(dataset_NCP)*0.85):], dataset_Normal[int(len(dataset_Normal)*0.85):]), axis=0)
# y_test = np.concatenate((CP_labels[int(len(dataset_CP)*0.85):], NCP_labels[int(len(dataset_NCP)*0.85):], Normal_labels[int(len(dataset_Normal)*0.85):]), axis=0)

#### 2 class
x_test = np.concatenate((dataset_CP_test, dataset_NCP_test), axis=0)
y_test = np.concatenate((CP_labels_test, NCP_labels_test), axis=0)
# x_test = np.concatenate((dataset_Normal_test, dataset_NCP_test), axis=0)
# y_test = np.concatenate((Normal_labels_test, NCP_labels_test), axis=0)

train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# AUTOTUNE = tf.data.experimental.AUTOTUNE
# GCS_PATH = KaggleDatasets().get_gcs_path()
BATCH_SIZE = 16#8#16 #* strategy.num_replicas_in_sync
IMAGE_SIZE = [160, 128, 32, 1] #[160, 128, 32]
EPOCHS = 50

train_ds = (
    train_loader.shuffle(len(x_train))
    .repeat()
    .batch(BATCH_SIZE)
)
# Only rescale.
val_ds = (
    validation_loader.shuffle(len(x_val))
    .repeat()
    .batch(BATCH_SIZE)
)

test_ds = (
    test_loader.batch(BATCH_SIZE)
)

print("data done")

# COUNT_NORMAL = len([filename for filename in train_filenames if "NORMAL" in filename])
# print("Normal images count in training set: " + str(COUNT_NORMAL))

# COUNT_PNEUMONIA = len([filename for filename in train_filenames if "PNEUMONIA" in filename])
# print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))

# weight_for_0 = (1 / COUNT_NORMAL)*(TRAIN_IMG_COUNT)/2.0 
# weight_for_1 = (1 / COUNT_PNEUMONIA)*(TRAIN_IMG_COUNT)/2.0

# class_weight = {0: weight_for_0, 1: weight_for_1}



# train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
# val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)

TRAIN_IMG_COUNT = len(x_train) #tf.data.experimental.cardinality(train_list_ds).numpy()
print("Training images count: " + str(TRAIN_IMG_COUNT))

VAL_IMG_COUNT = len(x_val) #tf.data.experimental.cardinality(val_list_ds).numpy()
print("Validating images count: " + str(VAL_IMG_COUNT))

# CLASS_NAMES = np.array([str(tf.strings.split(item, os.path.sep)[-1].numpy())[2:-1]
#                         for item in tf.io.gfile.glob(str(GCS_PATH + "/chest_xray/train/*"))])

# def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
#     # This is a small dataset, only load it once, and keep it in memory.
#     # use `.cache(filename)` to cache preprocessing work for datasets that don't
#     # fit in memory.
#     if cache:
#         if isinstance(cache, str):
#             ds = ds.cache(cache)
#         else:
#             ds = ds.cache()

#     ds = ds.shuffle(buffer_size=shuffle_buffer_size)

#     # Repeat forever
#     ds = ds.repeat()

#     ds = ds.batch(BATCH_SIZE)

#     # `prefetch` lets the dataset fetch batches in the background while the model
#     # is training.
#     ds = ds.prefetch(buffer_size=AUTOTUNE)

#     return ds

# train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# train_ds = prepare_for_training(train_ds)
# val_ds = prepare_for_training(val_ds)

# image_batch, label_batch = next(iter(train_ds))

def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.Conv3D(filters, 3, activation='relu', padding='same'), #SeparableConv2D
        tf.keras.layers.Conv3D(filters, 3, activation='relu', padding='same'), #SeparableConv2D
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool3D()
    ])
    return block

def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    return block

def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2], IMAGE_SIZE[3])),
        
        tf.keras.layers.Conv3D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv3D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool3D(),
        
        conv_block(32),
        conv_block(64),
        
        conv_block(128),
        tf.keras.layers.Dropout(0.2),
        
        conv_block(256),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),
        
        tf.keras.layers.Dense(3, activation='sigmoid') #2
    ])
    return model

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("3d_densenet_2class.h5",
                                                    save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.01, 20)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)

# model = build_model()
model = densenet3D((IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2], IMAGE_SIZE[3]), 2) # IMAGE_SIZE[3]
model.summary()

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

model.compile(
    optimizer='adam',
    loss='binary_crossentropy', #'categorical_crossentropy'
    metrics=METRICS
)

history = model.fit(
train_ds,
steps_per_epoch=TRAIN_IMG_COUNT // BATCH_SIZE,
epochs=EPOCHS,
validation_data=val_ds,
validation_steps=VAL_IMG_COUNT // BATCH_SIZE,
# class_weight=class_weight,
callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler]
)


fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["accuracy", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
plt.savefig('densenet_two_class_pneumonia_finetune_no_dup.png')

loss, acc, prec, rec = model.evaluate(test_ds)

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

print("done")
