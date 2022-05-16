import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# https://keras.io/examples/vision/3D_image_classification/#loading-data-and-preprocessing
loader_CP = np.load('../data-arrays-final/dataset_CP_train_2_scaled.npz')
loader_NCP = np.load('../data-arrays-final/dataset_NCP_train_2_scaled.npz')
loader_Normal = np.load('../data-arrays-final/dataset_Normal_train_2_scaled.npz')

dataset_CP = loader_CP['arr_0'] # 1176
dataset_NCP = loader_NCP['arr_0'] # 1280
dataset_Normal = loader_Normal['arr_0'] # 850

print('data loaded')

dataset_CP = dataset_CP.reshape(-1, 118, 160, 32)
dataset_NCP = dataset_NCP.reshape(-1, 118, 160, 32)
dataset_Normal = dataset_Normal.reshape(-1, 118, 160, 32)

CP_labels = np.array([[1,0,0] for _ in range(len(dataset_CP))])
NCP_labels = np.array([[0,1,0] for _ in range(len(dataset_NCP))])
Normal_labels = np.array([[0,0,1] for _ in range(len(dataset_Normal))])

x_train = np.concatenate((dataset_CP [:int(len(dataset_CP)*0.7)], dataset_NCP[:int(len(dataset_NCP)*0.7)], dataset_Normal[:int(len(dataset_Normal)*0.7)]), axis=0)
y_train = np.concatenate((CP_labels[:int(len(dataset_CP)*0.7)], NCP_labels[:int(len(dataset_NCP)*0.7)], Normal_labels[:int(len(dataset_Normal)*0.7)]), axis=0)
x_val = np.concatenate((dataset_CP[int(len(dataset_CP)*0.7):], dataset_NCP[int(len(dataset_NCP)*0.7):], dataset_Normal[int(len(dataset_Normal)*0.7):]), axis=0)
y_val = np.concatenate((CP_labels[int(len(dataset_CP)*0.7):], NCP_labels[int(len(dataset_NCP)*0.7):], Normal_labels[int(len(dataset_Normal)*0.7):]), axis=0)

train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

print('input created')

batch_size = 2
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

def get_model(width=128, height=128, depth=50):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth))

    x = layers.Conv2D(filters=8, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool2D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool2D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool2D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool2D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    # x = layers.Flatten(x)
    outputs = layers.Dense(units=3, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="2dcnn")
    return model


# Build model.
model = get_model(width=118, height=160, depth=32)
model.summary()

# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 30
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
plt.savefig('my_plot.png')

print('done')
