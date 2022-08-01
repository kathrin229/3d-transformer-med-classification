import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

from dense_net_2D import densenet
from dense_net_3D import densenet3D

start_time = time.time()

# loader_CP = np.load('./data-arrays/dataset_CP_train_5_corrected.npz')
# loader_NCP = np.load('./data-arrays/dataset_NCP_train_5_corrected.npz')
# loader_Normal = np.load('./data-arrays/dataset_Normal_train_5_corrected.npz')

# loader_CP_valid = np.load('./data-arrays/dataset_CP_valid_5_corrected.npz')
# loader_NCP_valid = np.load('./data-arrays/dataset_NCP_valid_5_corrected.npz')
# loader_Normal_valid = np.load('./data-arrays/dataset_Normal_valid_5_corrected.npz')

# loader_CP_test = np.load('./data-arrays/dataset_CP_test_5_corrected.npz')
# loader_NCP_test = np.load('./data-arrays/dataset_NCP_test_5_corrected.npz')
# loader_Normal_test = np.load('./data-arrays/dataset_Normal_test_5_corrected.npz')

loader_CP = np.load('./data-arrays/dataset_CP_train_small.npz')
loader_NCP = np.load('./data-arrays/dataset_NCP_train_small.npz')
loader_Normal = np.load('./data-arrays/dataset_Normal_train_small.npz')

loader_CP_valid = np.load('./data-arrays/dataset_CP_valid_small.npz')
loader_NCP_valid = np.load('./data-arrays/dataset_NCP_valid_small.npz')
loader_Normal_valid = np.load('./data-arrays/dataset_Normal_valid_small.npz')

loader_CP_test = np.load('./data-arrays/dataset_CP_test_small.npz')
loader_NCP_test = np.load('./data-arrays/dataset_NCP_test_small.npz')
loader_Normal_test = np.load('./data-arrays/dataset_Normal_test_small.npz')

dataset_CP = loader_CP['arr_0']
dataset_NCP = loader_NCP['arr_0']
dataset_Normal = loader_Normal['arr_0']

dataset_CP_valid = loader_CP_valid['arr_0']
dataset_NCP_valid = loader_NCP_valid['arr_0']
dataset_Normal_valid = loader_Normal_valid['arr_0']

dataset_CP_test = loader_CP_test['arr_0']
dataset_NCP_test = loader_NCP_test['arr_0']
dataset_Normal_test = loader_Normal_test['arr_0']

# dataset_CP = dataset_CP.reshape(-1, 160, 128, 32)
# dataset_NCP = dataset_NCP.reshape(-1, 160, 128, 32)
# dataset_Normal = dataset_Normal.reshape(-1, 160, 128, 32)

# dataset_CP_valid = dataset_CP_valid.reshape(-1, 160, 128, 32)
# dataset_NCP_valid = dataset_NCP_valid.reshape(-1, 160, 128, 32)
# dataset_Normal_valid = dataset_Normal_valid.reshape(-1, 160, 128, 32)

# dataset_CP_test = dataset_CP_test.reshape(-1, 160, 128, 32)
# dataset_NCP_test = dataset_NCP_test.reshape(-1, 160, 128, 32)
# dataset_Normal_test = dataset_Normal_test.reshape(-1, 160, 128, 32)

dataset_CP = dataset_CP.reshape(-1, 80, 64, 16)
dataset_NCP = dataset_NCP.reshape(-1, 80, 64, 16)
dataset_Normal = dataset_Normal.reshape(-1, 80, 64, 16)

dataset_CP_valid = dataset_CP_valid.reshape(-1, 80, 64, 16)
dataset_NCP_valid = dataset_NCP_valid.reshape(-1, 80, 64, 16)
dataset_Normal_valid = dataset_Normal_valid.reshape(-1, 80, 64, 16)

dataset_CP_test = dataset_CP_test.reshape(-1, 80, 64, 16)
dataset_NCP_test = dataset_NCP_test.reshape(-1, 80, 64, 16)
dataset_Normal_test = dataset_Normal_test.reshape(-1, 80, 64, 16)


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

#### 2 class
x_test = np.concatenate((dataset_CP_test, dataset_NCP_test), axis=0)
y_test = np.concatenate((CP_labels_test, NCP_labels_test), axis=0)
# x_test = np.concatenate((dataset_Normal_test, dataset_NCP_test), axis=0)
# y_test = np.concatenate((Normal_labels_test, NCP_labels_test), axis=0)

train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

BATCH_SIZE = 16
IMAGE_SIZE = [80, 64, 16, 1] #[160, 128, 32, 1]
EPOCHS = 50

train_ds = (
    train_loader.shuffle(len(x_train))
    .repeat()
    .batch(BATCH_SIZE)
)

val_ds = (
    validation_loader.shuffle(len(x_val))
    .repeat()
    .batch(BATCH_SIZE)
)

test_ds = (
    test_loader.batch(BATCH_SIZE)
)

print("data done")


TRAIN_IMG_COUNT = len(x_train)
print("Training images count: " + str(TRAIN_IMG_COUNT))

VAL_IMG_COUNT = len(x_val)
print("Validating images count: " + str(VAL_IMG_COUNT))


checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("3d_densenet_2class_small_data_changed.h5",
                                                    save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.01, 20)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)

model = densenet3D((IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2], IMAGE_SIZE[3]), 2) # IMAGE_SIZE[3] # densenet 3D
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

print("--- %s seconds ---" % (time.time() - start_time))

fit_time = time.time()

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["accuracy", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
plt.savefig('densenet_two_class_pneumonia_3D.png')

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

print("--- %s seconds ---" % (time.time() - fit_time))

print("done")
