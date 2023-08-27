from medmnist import FractureMNIST3D, VesselMNIST3D, SynapseMNIST3D, AdrenalMNIST3D, NoduleMNIST3D, OrganMNIST3D
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset
# import tensorflow as tf


def get_num_classes(medmnist3d_dataset):
    dataset_train = []
    if medmnist3d_dataset == "FractureMNIST3D":
        dataset_train = FractureMNIST3D(split="train", download=True)
    elif medmnist3d_dataset == "VesselMNIST3D":
        dataset_train = VesselMNIST3D(split="train", download=True)
    elif medmnist3d_dataset == "SynapseMNIST3D":
        dataset_train = SynapseMNIST3D(split="train", download=True)
    elif medmnist3d_dataset == "AdrenalMNIST3D":
        dataset_train = AdrenalMNIST3D(split="train", download=True)
    elif medmnist3d_dataset == "NoduleMNIST3D":
        dataset_train = NoduleMNIST3D(split="train", download=True)
    elif medmnist3d_dataset == "OrganMNIST3D":
        dataset_train = OrganMNIST3D(split="train", download=True)
   
    return len(dataset_train.info['label'])


def plot_dataset_instance(medmnist3d_dataset):
    dataset_train = []
    if medmnist3d_dataset == "FractureMNIST3D":
        dataset_train = FractureMNIST3D(split="train", download=True)
    elif medmnist3d_dataset == "VesselMNIST3D":
        dataset_train = VesselMNIST3D(split="train", download=True)
    elif medmnist3d_dataset == "SynapseMNIST3D":
        dataset_train = SynapseMNIST3D(split="train", download=True)
    elif medmnist3d_dataset == "AdrenalMNIST3D":
        dataset_train = AdrenalMNIST3D(split="train", download=True)
    elif medmnist3d_dataset == "NoduleMNIST3D":
        dataset_train = NoduleMNIST3D(split="train", download=True)
    elif medmnist3d_dataset == "OrganMNIST3D":
        dataset_train = OrganMNIST3D(split="train", download=True)

    numbers = [1]
    for number in numbers:
        fig = plt.figure(figsize=(12, 12))
        columns = 7
        rows = 4
        scan = dataset_train.imgs[number]
        i = 1
        for img in scan:
            img = img * 255
            fig.add_subplot(rows, columns, i)
            plt.imshow(img, cmap='gray')
            i += 1
    plt.savefig("FractureMnist3D.png")
    # plt.show()
    print('done')


def load_dataset_train_valid(medmnist3d_dataset):
    """
    Loading training and validation dataset for TimeSformer (pytorch)
    Args:
        medmnist3d_dataset (str): specifying which of the six mnist 3d datasets should be used
    Returns:
        train_dataset (torch.TensorDataset): the dataset for training
        val_dataset (torch.TensorDataset): the dataset for validation
    """
    dataset_train = []
    dataset_test = []
    if medmnist3d_dataset == "FractureMNIST3D":
        dataset_train = FractureMNIST3D(split="train", download=True)
        dataset_test = FractureMNIST3D(split="test", download=True)
    elif medmnist3d_dataset == "VesselMNIST3D":
        dataset_train = VesselMNIST3D(split="train", download=True)
        dataset_test = VesselMNIST3D(split="test", download=True)
    elif medmnist3d_dataset == "SynapseMNIST3D":
        dataset_train = SynapseMNIST3D(split="train", download=True)
        dataset_test = SynapseMNIST3D(split="test", download=True)
    elif medmnist3d_dataset == "AdrenalMNIST3D":
        dataset_train = AdrenalMNIST3D(split="train", download=True)
        dataset_test = AdrenalMNIST3D(split="test", download=True)
    elif medmnist3d_dataset == "NoduleMNIST3D":
        dataset_train = NoduleMNIST3D(split="train", download=True)
        dataset_test = NoduleMNIST3D(split="test", download=True)
    elif medmnist3d_dataset == "OrganMNIST3D":
        dataset_train = OrganMNIST3D(split="train", download=True)
        dataset_test = OrganMNIST3D(split="test", download=True)

    # scaling data between 0 and 1
    dataset_train.imgs = dataset_train.imgs / 255

    # making data fit for timesformer input: (B x C x H x W x D)
    dataset_train.imgs = dataset_train.imgs[:, :, :, :, np.newaxis]
    dataset_train.imgs = dataset_train.imgs.reshape(-1, 1, 28, 28, 28)

    # determine validation data set size (20%)
    total_size = len(dataset_train.imgs) + len(dataset_test.imgs)
    valid_size = int(total_size * 0.2)
    train_size = len(dataset_train.imgs) - valid_size

    x_train = dataset_train.imgs[:train_size]
    y_train = np.squeeze(dataset_train.labels[:train_size])
    x_valid = dataset_train.imgs[train_size:]
    y_valid = np.squeeze(dataset_train.labels[train_size:])

    tensor_x_train = torch.Tensor(x_train)
    tensor_y_train = torch.Tensor(y_train)
    tensor_x_val = torch.Tensor(x_valid)
    tensor_y_val = torch.Tensor(y_valid)

    train_dataset = TensorDataset(tensor_x_train,tensor_y_train)
    val_dataset = TensorDataset(tensor_x_val,tensor_y_val)

    return train_dataset, val_dataset


def load_dataset_test(medmnist3d_dataset):
    """
    Loading test dataset for TimeSformer (pytorch)
    Args:
        medmnist3d_dataset (str): specifying which of the six mnist 3d datasets should be used
    Returns:
        test_dataset (torch.TensorDataset): the dataset for testing
    """
    dataset_test = []
    if medmnist3d_dataset == "FractureMNIST3D":
        dataset_test = FractureMNIST3D(split="test", download=True)
    elif medmnist3d_dataset == "VesselMNIST3D":
        dataset_test = VesselMNIST3D(split="test", download=True)
    elif medmnist3d_dataset == "SynapseMNIST3D":
        dataset_test = SynapseMNIST3D(split="test", download=True)
    elif medmnist3d_dataset == "AdrenalMNIST3D":
        dataset_test = AdrenalMNIST3D(split="test", download=True)
    elif medmnist3d_dataset == "NoduleMNIST3D":
        dataset_test = NoduleMNIST3D(split="test", download=True)
    elif medmnist3d_dataset == "OrganMNIST3D":
        dataset_test = OrganMNIST3D(split="test", download=True)

    # scaling data between 0 and 1
    dataset_test.imgs = dataset_test.imgs / 255

    # making data fit for timesformer input: (B x C x H x W x D)
    dataset_test.imgs = dataset_test.imgs[:, :, :, :, np.newaxis]
    dataset_test.imgs = dataset_test.imgs.reshape(-1, 1, 28, 28, 28)

    x_test = dataset_test.imgs
    y_test = np.squeeze(dataset_test.labels)

    tensor_x_test = torch.Tensor(x_test)
    tensor_y_test = torch.Tensor(y_test)

    test_dataset = TensorDataset(tensor_x_test,tensor_y_test)

    return test_dataset