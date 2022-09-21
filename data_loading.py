import numpy as np
# import torch
# from torch.utils.data import TensorDataset

import tensorflow as tf

def load_dataset_train_valid(size, classes):
    # sizes: '160x128x32' '80x64x16-2nd' '80x64x16-mid'
    # classes: ['CP', 'NCP', 'Normal']

    data_train_list = []
    data_valid_list = []
    label_train_list = []
    label_valid_list = []
    
    for i, type in enumerate(classes):
        loader_train = np.load('Data/dataset_'+type+'_train_'+size+'.npz')
        loader_valid = np.load('Data/dataset_'+type+'_valid_'+size+'.npz')

        dataset_train = loader_train['arr_0'] 
        dataset_valid = loader_valid['arr_0']

        if size in ['160x128x32']:
            dataset_train = dataset_train.reshape(-1, 160, 128, 32)
            dataset_train = dataset_train[:, :, :, :, np.newaxis]
            dataset_train = dataset_train.reshape(-1, 1, 32, 128, 160)

            dataset_valid = dataset_valid.reshape(-1, 160, 128, 32)
            dataset_valid = dataset_valid[:, :, :, :, np.newaxis]
            dataset_valid = dataset_valid.reshape(-1, 1, 32, 128, 160)

        elif size in ['80x64x16-2nd', '80x64x16-mid']:
            dataset_train = dataset_train.reshape(-1, 80, 64, 16)
            dataset_train = dataset_train[:, :, :, :, np.newaxis]
            dataset_train = dataset_train.reshape(-1, 1, 16, 64, 80)

            dataset_valid = dataset_valid.reshape(-1, 80, 64, 16)
            dataset_valid = dataset_valid[:, :, :, :, np.newaxis]
            dataset_valid = dataset_valid.reshape(-1, 1, 16, 64, 80)


        labels_train = np.array([i for _ in range(len(dataset_train))])
        labels_valid = np.array([i for _ in range(len(dataset_valid))])

        data_train_list.append(dataset_train)
        data_valid_list.append(dataset_valid)
        label_train_list.append(labels_train)
        label_valid_list.append(labels_valid)

    x_train = np.concatenate(data_train_list, axis=0)
    y_train = np.concatenate(label_train_list, axis=0)
    x_val = np.concatenate(data_valid_list, axis=0)
    y_val = np.concatenate(label_valid_list, axis=0)


    tensor_x_train = torch.Tensor(x_train)
    tensor_y_train = torch.Tensor(y_train)
    tensor_x_val = torch.Tensor(x_val)
    tensor_y_val = torch.Tensor(y_val)

    train_dataset = TensorDataset(tensor_x_train,tensor_y_train)
    val_dataset = TensorDataset(tensor_x_val,tensor_y_val)

    return train_dataset, val_dataset


def load_dataset_test(size, classes):
    # sizes: '160x128x32' '80x64x16-2nd' '80x64x16-mid'
    # classes: ['CP', 'NCP', 'Normal']

    data_test_list = []
    label_test_list = []
    
    for i, type in enumerate(classes):
        loader_test = np.load('Data/dataset_'+type+'_test_'+size+'.npz')

        dataset_test = loader_test['arr_0'] 

        if size in ['160x128x32']:
            dataset_test = dataset_test.reshape(-1, 160, 128, 32)
            dataset_test = dataset_test[:, :, :, :, np.newaxis]
            dataset_test = dataset_test.reshape(-1, 1, 32, 128, 160)
            dataset_test = dataset_test.reshape(-1, 1, 1, 32, 128, 160)

        elif size in ['80x64x16-2nd', '80x64x16-mid']:
            dataset_test = dataset_test.reshape(-1, 80, 64, 16)
            dataset_test = dataset_test[:, :, :, :, np.newaxis]
            dataset_test = dataset_test.reshape(-1, 1, 16, 64, 80)
            dataset_test = dataset_test.reshape(-1, 1, 1, 16, 64, 80)

        labels_test = np.array([i for _ in range(len(dataset_test))])

        data_test_list.append(dataset_test)
        label_test_list.append(labels_test)

    x_test = np.concatenate(data_test_list, axis=0)
    y_test = np.concatenate(label_test_list, axis=0)

    tensor_x_test = torch.Tensor(x_test)
    tensor_y_test = torch.Tensor(y_test)

    test_dataset = TensorDataset(tensor_x_test,tensor_y_test)

    return test_dataset
        

def load_dataset_train_valid_test(size, classes):
    data_train_list = []
    data_valid_list = []
    data_test_list = []
    
    label_train_list = []
    label_valid_list = []
    label_test_list = []

    label2class =  [[1,0], [0,1]]
    label3class = [[1,0,0], [0,1,0], [0,0,1]]
    
    for i, type in enumerate(classes):
        loader_train = np.load('Data/dataset_'+type+'_train_'+size+'.npz')
        loader_valid = np.load('Data/dataset_'+type+'_valid_'+size+'.npz')
        loader_test = np.load('Data/dataset_'+type+'_test_'+size+'.npz')

        dataset_train = loader_train['arr_0'] 
        dataset_valid = loader_valid['arr_0']
        dataset_test = loader_test['arr_0']

        if size in ['160x128x32']:
            dataset_train = dataset_train.reshape(-1, 160, 128, 32)
            dataset_valid = dataset_valid.reshape(-1, 160, 128, 32)
            dataset_test = dataset_test.reshape(-1, 160, 128, 32)

            if len(classes) == 3:
                dataset_train = dataset_train[:, :, :, :, np.newaxis]
                dataset_valid = dataset_valid[:, :, :, :, np.newaxis]
                dataset_test = dataset_test[:, :, :, :, np.newaxis]

        elif size in ['80x64x16-2nd', '80x64x16-mid']:
            dataset_train = dataset_train.reshape(-1, 80, 64, 16)
            dataset_valid = dataset_valid.reshape(-1, 80, 64, 16)
            dataset_test = dataset_test.reshape(-1, 80, 64, 16)

            if len(classes) == 3:
                dataset_train = dataset_train[:, :, :, :, np.newaxis]
                dataset_valid = dataset_valid[:, :, :, :, np.newaxis]
                dataset_test = dataset_test[:, :, :, :, np.newaxis]
        
        if len(classes) == 3:
            labels_train = np.array([label3class[i] for _ in range(len(dataset_train))])
            labels_valid = np.array([label3class[i] for _ in range(len(dataset_valid))])
            labels_test = np.array([label3class[i] for _ in range(len(dataset_test))])
        else:
            labels_train = np.array([label2class[i] for _ in range(len(dataset_train))])
            labels_valid = np.array([label2class[i] for _ in range(len(dataset_valid))])
            labels_test = np.array([label2class[i] for _ in range(len(dataset_test))])

        data_train_list.append(dataset_train)
        data_valid_list.append(dataset_valid)
        label_train_list.append(labels_train)
        label_valid_list.append(labels_valid)
        data_test_list.append(dataset_test)
        label_test_list.append(labels_test)

    x_train = np.concatenate(data_train_list, axis=0)
    y_train = np.concatenate(label_train_list, axis=0)
    x_val = np.concatenate(data_valid_list, axis=0)
    y_val = np.concatenate(label_valid_list, axis=0)
    x_test = np.concatenate(data_test_list, axis=0)
    y_test = np.concatenate(label_test_list, axis=0)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return train_dataset, val_dataset, test_dataset, x_train, x_val, x_test, y_train, y_val, y_test
