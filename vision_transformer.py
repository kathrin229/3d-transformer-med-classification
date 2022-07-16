import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras import layers

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
CP_labels = np.array([[1,0,0] for _ in range(len(dataset_CP))])
NCP_labels = np.array([[0,1,0] for _ in range(len(dataset_NCP))])
Normal_labels = np.array([[0,0,1] for _ in range(len(dataset_Normal))])

#### 3 class
x_train = np.concatenate((dataset_CP, dataset_NCP, dataset_Normal), axis=0)
y_train = np.concatenate((CP_labels, NCP_labels, Normal_labels), axis=0)

#### 3 class
CP_labels_valid = np.array([[1,0,0] for _ in range(len(dataset_CP_valid))])
NCP_labels_valid = np.array([[0,1,0] for _ in range(len(dataset_NCP_valid))])
Normal_labels_valid = np.array([[0,0,1] for _ in range(len(dataset_Normal_valid))])

#### 3 class
x_val = np.concatenate((dataset_CP_valid, dataset_NCP_valid, dataset_Normal_valid), axis=0)
y_val = np.concatenate((CP_labels_valid, NCP_labels_valid, Normal_labels_valid), axis=0)

#### 3 class
CP_labels_test = np.array([[1,0,0] for _ in range(len(dataset_CP_test))])
NCP_labels_test = np.array([[0,1,0] for _ in range(len(dataset_NCP_test))])
Normal_labels_test = np.array([[0,0,1] for _ in range(len(dataset_Normal_test))])

#### 3 class
x_test = np.concatenate((dataset_CP_test, dataset_NCP_test, dataset_Normal_test), axis=0)
y_test = np.concatenate((CP_labels_test, NCP_labels_test, Normal_labels_test), axis=0)

train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

BATCH_SIZE = 64 #8 #16
IMAGE_SIZE = [160, 128, 32, 1] #[160, 128, 32]
EPOCHS = 50

train_ds = (
    train_loader.shuffle(len(x_train))
    # .repeat()
    .batch(BATCH_SIZE)
)
# Only rescale.
val_ds = (
    validation_loader.shuffle(len(x_val))
    # .repeat()
    .batch(BATCH_SIZE)
)

# test_ds = (
#     test_loader.batch(BATCH_SIZE)
# )

print("data done")


class generate_patch(layers.Layer):
  def __init__(self, patch_size):
    super(generate_patch, self).__init__()
    self.patch_size = patch_size
    
  def call(self, images):
    batch_size = tf.shape(images)[0]
    patches = layers.Conv3D(self.patch_size*self.patch_size*32*1, (self.patch_size, self.patch_size, 32), (self.patch_size, self.patch_size, 32), padding='valid')(images)

    patches = tf.extract_volume_patches(input=images, 
                                            ksizes=[1, self.patch_size, self.patch_size, 32, 1], 
                                            strides=[1, self.patch_size, self.patch_size, 32, 1], padding="VALID")
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims]) #here shape is (batch_size, num_patches, patch_h*patch_w*c) 
    return patches

from itertools import islice, count

train_iter_7im, train_iter_7label = next(islice(train_loader, 7, None)) # access the 7th element from the iterator



train_iter_7im = tf.expand_dims(train_iter_7im, 0)
train_iter_7label = train_iter_7label.numpy()

print('check shapes: ', train_iter_7im.shape) 

class_types = ['CP', 'NCP', 'Normal']

patch_size= 16 
######################
# num patches (W * H) /P^2 where W, H are from original image, P is patch dim. 
# Original image (H * W * C), patch N * P*P *C, N num patches
######################
# generate_patch_layer = generate_patch(patch_size=patch_size)
# patches = generate_patch_layer(train_iter_7im)

# print ('patch per image and patches shape: ', patches.shape[1], '\n', patches.shape)


##########-----------
def render_image_and_patches(image, patches):
    plt.figure(figsize=(6, 6))
    image = tf.reshape(image, [1, 32, 128, 160, 1])
    image = image * 255 # rescale from scaling between 0 and 1 to 0 and 255
    plt.imshow(tf.cast(image[0][0], tf.uint8)) #x
    plt.savefig('blocks.png')
    # plt.xlabel(class_types [np.argmax(train_iter_7label)], fontsize=13)
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(6, 6))
    #plt.suptitle(f"Image Patches", size=13)
    # patches = tf.reshape(patches, [320, 8, 16, 16])
    # for i, patch in enumerate(patches[0]):
    #     ax = plt.subplot(2, 4, i+1)
    #     # patch_img = tf.reshape(patch, (patch_size, patch_size, 8, 1))
    #     # patch_img = tf.reshape(patch, (8, patch_size, patch_size, 1))
    #     patch = patch * 255
    #     ax.imshow(tf.cast(patch, tf.uint8))
    #     ax.axis('off')    
    # plt.savefig('patches.png')
    patches_iterate = patches #tf.reshape(patches, (1, 4, 20, 8192))#(1, 4, 80, 2048))
    for i, patch in enumerate(patches_iterate[0]): # patches_iterate[0][0]
        ax = plt.subplot(8, 10, i+1) # 8, 10
        # patch_img = tf.reshape(patch, (patch_size, patch_size, 8, 1))
        patch_img = tf.reshape(patch, (32, patch_size, patch_size, 1))
        patch_img = patch_img * 255
        ax.imshow(patch_img[0].numpy().astype("uint8")) #x
        ax.axis('off')    
    plt.savefig('patches.png')

# render_image_and_patches(train_iter_7im, patches)

class PatchEncode_Embed(layers.Layer):
  '''
  2 steps happen here
  1. flatten the patches 
  2. Map to dim D; patch embeddings  
  '''
  def __init__(self, num_patches, projection_dim):
    super(PatchEncode_Embed, self).__init__()
    self.num_patches = num_patches
    self.projection = layers.Dense(units=projection_dim)# activation = linear
    self.position_embedding = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim)
    
  def call(self, patch):
    positions = tf.range(start=0, limit=self.num_patches, delta=1)
    encoded = self.projection(patch) + self.position_embedding(positions)
    return encoded

# patch_encoder = PatchEncode_Embed(80, 80)(patches)
# print (tf.shape(patch_encoder))

def generate_patch_conv_orgPaper_f(patch_size, hidden_size, inputs):
  patches = layers.Conv3D(patch_size*patch_size*32*1, (patch_size, patch_size, 32), (patch_size, patch_size, 32), padding='valid')(inputs)
  # patches = layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
  row_axis, col_axis = (1, 2) # channels last images
  seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
  x = tf.reshape(patches, [-1, seq_len, hidden_size])
  return x


def mlp_block_f(mlp_dim, inputs):
  x = layers.Dense(units=mlp_dim, activation=tf.nn.gelu)(inputs)
  x = layers.Dropout(rate=0.1)(x) # dropout rate is from original paper,
  x = layers.Dense(units=inputs.shape[-1], activation=tf.nn.gelu)(x)
  x = layers.Dropout(rate=0.1)(x)
  return x

def Encoder1Dblock_f(num_heads, mlp_dim, inputs):
  x = layers.LayerNormalization(dtype=inputs.dtype)(inputs)
  x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=0.1)(x, x) # self attention multi-head, dropout_rate is from original implementation
  x = layers.Add()([x, inputs]) # 1st residual part 
  
  y = layers.LayerNormalization(dtype=x.dtype)(x)
  y = mlp_block_f(mlp_dim, y)
  y_1 = layers.Add()([y, x]) #2nd residual part 
  return y_1

class AddPositionEmbs(layers.Layer):
  """Adds (optionally learned) positional embeddings to the inputs."""

  def __init__(self, posemb_init=None, **kwargs):
    super().__init__(**kwargs)
    self.posemb_init = posemb_init
    #posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input') # used in original code

  def build(self, inputs_shape):
    pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
    self.pos_embedding = self.add_weight('pos_embedding', pos_emb_shape, initializer=self.posemb_init)

  def call(self, inputs, inputs_positions=None):
    # inputs.shape is (batch_size, seq_len, emb_dim).
    pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)

    return inputs + pos_embedding

def Encoder_f(num_layers, mlp_dim, num_heads, inputs):
  x = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input')(inputs)
  x = layers.Dropout(rate=0.2)(x)
  for _ in range(num_layers):
    x = Encoder1Dblock_f(num_heads, mlp_dim, x)

  encoded = layers.LayerNormalization(name='encoder_norm')(x)
  return encoded

######################################
# hyperparameter section 
###################################### 
transformer_layers = 2
patch_size = 16 #16 #4 
hidden_size = 8192 #8192 #512
num_heads = 4
mlp_dim = 128

######################################

rescale_layer = tf.keras.Sequential([layers.experimental.preprocessing.Rescaling(1./255)])


def build_ViT():
  inputs = layers.Input(shape= (160, 128, 32, 1))#x_train.shape[1:])
  # rescaling (normalizing pixel val between 0 and 1)
  # rescale = rescale_layer(inputs)

  # generate patches with conv layer
  patches = generate_patch_conv_orgPaper_f(patch_size, hidden_size, inputs)

  ######################################
  # ready for the transformer blocks
  ######################################
  encoder_out = Encoder_f(transformer_layers, mlp_dim, num_heads, patches)  

  #####################################
  #  final part (mlp to classification)
  #####################################
  #encoder_out_rank = int(tf.experimental.numpy.ndim(encoder_out))
  im_representation = tf.reduce_mean(encoder_out, axis=1)  # (1,) or (1,2)
  # similar to the GAP, this is from original Google GitHub

  logits = layers.Dense(units=len(class_types), name='head', kernel_initializer=tf.keras.initializers.zeros)(im_representation) # !!! important !!! activation is linear 

  final_model = tf.keras.Model(inputs = inputs, outputs = logits)
  return final_model

ViT_model = build_ViT()
ViT_model.summary()

ViT_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3), 
                  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"), tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5 acc')]) 
#tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")],) 
# from logits = True, because Dense layer has linear activation


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                              patience=5, min_lr=1e-5, verbose=1)


ViT_Train = ViT_model.fit(train_ds, 
                        epochs = 5, 
                        validation_data=val_ds, callbacks=[reduce_lr])



### Plot train and validation curves
loss = ViT_Train.history['loss']
v_loss = ViT_Train.history['val_loss']

acc = ViT_Train.history['accuracy'] 
v_acc = ViT_Train.history['val_accuracy']

top5_acc = ViT_Train.history['top5 acc']
val_top5_acc = ViT_Train.history['val_top5 acc']
epochs = range(len(loss))

fig = plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.yscale('log')
plt.plot(epochs, loss, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Loss')
plt.plot(epochs, v_loss, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Loss')
# plt.ylim(0.3, 100)
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.subplot(1, 3, 2)
plt.plot(epochs, acc, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Acc')
plt.plot(epochs, v_acc, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Acc') 
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.subplot(1, 3, 3)
plt.plot(epochs, top5_acc, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Top 5 Acc')
plt.plot(epochs, val_top5_acc, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Top5 Acc') 
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Top5 Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('train_acc.png', dpi=250)
# plt.show()

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def conf_matrix(predictions): 
    ''' Plots conf. matrix and classification report '''
    rounded_labels=np.argmax(y_test, axis=1)
    cm=confusion_matrix(rounded_labels, np.argmax(np.round(predictions), axis=1))
    # cm=confusion_matrix(y_test, np.argmax(np.round(predictions), axis=1))
    print("Classification Report:\n")
    cr=classification_report(y_test,
                                np.argmax(np.round(predictions), axis=1), 
                                target_names=[class_types[i] for i in range(len(class_types))])
    print(cr)
    plt.figure(figsize=(12,12))
    sns_hmp = sns.heatmap(cm, annot=True, xticklabels = [class_types[i] for i in range(len(class_types))], 
                yticklabels = [class_types[i] for i in range(len(class_types))], fmt="d")
    fig = sns_hmp.get_figure()
    plt.savefig('conf_matrix.png', dpi=250)

pred_class_resnet50 = ViT_model.predict(x_test)

conf_matrix(pred_class_resnet50)

print('done')