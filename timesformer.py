from pathlib import Path

import torch
from TimeSformer.timesformer.models.vit import TimeSformer
from torch.utils.data import TensorDataset, DataLoader
import copy
import numpy as np

model_file = Path.home()/'3d-transformer-med-classification/TimeSformer_divST_8x32_224_K600-2.pyth'
model_file.exists()

model = TimeSformer(img_size=224, num_classes=600, num_frames=32, attention_type='divided_space_time',  pretrained_model=str(model_file))
# model.model.default_cfg['input_size'] = (1, 224, 224)

model2 = copy.deepcopy(model)

weight = model.model.patch_embed.proj.weight
model2.model.patch_embed.proj = torch.nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
weight_new = torch.reshape(weight, (3, 768, 16, 16))
weight_new_reshape = torch.reshape(weight_new[0], (768, 1, 16, 16))
with torch.no_grad():
    model2.model.patch_embed.proj.weight = torch.nn.Parameter(weight_new_reshape)
model2.model.head = torch.nn.Linear(in_features = 768, out_features=3, bias = True)

# dummy_video = torch.randn(2, 1, 32, 128, 160) # (batch x channels x frames x height x width)

# pred = model2(dummy_video,) # (2, 600)

# assert pred.shape == (2,600)

# Todo: load data, train loop, eval
loader_CP = np.load('./data-arrays/dataset_CP_train_5_corrected.npz')
loader_NCP = np.load('./data-arrays/dataset_NCP_train_5_corrected.npz')
loader_Normal = np.load('./data-arrays/dataset_Normal_train_5_corrected.npz')

dataset_CP = loader_CP['arr_0'] # 1176
dataset_NCP = loader_NCP['arr_0'] # 1280
dataset_Normal = loader_Normal['arr_0'] # 850

dataset_CP = dataset_CP.reshape(-1, 160, 128, 32)
dataset_NCP = dataset_NCP.reshape(-1, 160, 128, 32)
dataset_Normal = dataset_Normal.reshape(-1, 160, 128, 32)

dataset_CP = dataset_CP[:, :, :, :, np.newaxis]
dataset_NCP = dataset_NCP[:, :, :, :, np.newaxis]
dataset_Normal = dataset_Normal[:, :, :, :, np.newaxis]

dataset_CP = dataset_CP.reshape(-1, 1, 32, 128, 160)
dataset_NCP = dataset_NCP.reshape(-1, 1, 32, 128, 160)
dataset_Normal = dataset_Normal.reshape(-1, 1, 32, 128, 160)

#### 3 class
CP_labels = np.array([[1,0,0] for _ in range(len(dataset_CP))])
NCP_labels = np.array([[0,1,0] for _ in range(len(dataset_NCP))])
Normal_labels = np.array([[0,0,1] for _ in range(len(dataset_Normal))])

#### 3 class
x_train = np.concatenate((dataset_CP, dataset_NCP, dataset_Normal), axis=0)
y_train = np.concatenate((CP_labels, NCP_labels, Normal_labels), axis=0)

tensor_x = torch.Tensor(x_train) # transform to torch tensor
tensor_y = torch.Tensor(y_train)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
trainloader = DataLoader(my_dataset, batch_size=4, shuffle=True, num_workers=2) # create your dataloader
print("data finished")
import torch.optim as optim

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model2(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.7f}')
            running_loss = 0.0

print('Finished Training')

PATH = './timesformer_net.pth'
torch.save(model2.state_dict(), PATH)

print('done')