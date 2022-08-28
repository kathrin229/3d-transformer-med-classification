import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

from dense_net_2D import densenet
from dense_net_3D import densenet3D

from data_loading import load_dataset_train_valid_test

start_time = time.time()

size = 'small_middle_part'
framework = 'tensorflow'
classes = ['CP', 'NCP']
n_classes = 2
loss = 'binary_crossentropy' #'categorical_crossentropy' #

train_loader, validation_loader, test_loader, x_train, x_val, x_test, y_train, y_val, y_test = load_dataset_train_valid_test(size, framework, classes)
# test_loader, x_test, y_test = load_dataset_test(size, framework, classes)

model_type = 'densenet3D' # 'densenet3D
model_name = "3d_densenet_2class_small_middle_batch_8"

patience_early_stopping = 15
BATCH_SIZE = 8
IMAGE_SIZE = [80, 64, 16, 1] #[160, 128, 32, 1] #
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


checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_name,
                                                    save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience_early_stopping,
                                                     restore_best_weights=True)

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.01, 20)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)

if model_type in ['densenet3D']:
    model = densenet3D((IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2], IMAGE_SIZE[3]), n_classes)
elif model_type in ['densenet2D']:
    model = densenet((IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), n_classes)
# model.summary()

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

model.compile(
    optimizer='adam',
    loss=loss,
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
plt.savefig(model_name+'.png')

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

print("Training--- %s seconds ---" % (fit_time - start_time))
print("Total--- %s seconds ---" % (time.time() - start_time))

print("done")
