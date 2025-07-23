import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class GMSSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Завантажуємо ваші дані
        # Переконайтеся, що ви зберегли ваші файли саме з цими назвами у папці data_path (dataset/gms)
        try:
            train_data_raw = np.load(os.path.join(data_path, "game_train.npy"))
            test_data_raw = np.load(os.path.join(data_path, "game_test.npy"))
            test_labels_raw = np.load(os.path.join(data_path, "game_test_label.npy"))
            # Якщо у вас є окремий файл міток для тренування, завантажте його:
            # train_labels_raw = np.load(os.path.join(data_path, "game_train_label.npy"))
        except FileNotFoundError as e:
            print(f"Помилка завантаження файлів даних для GMS: {e}")
            print(f"Перевірте, чи файли game_train.npy, game_test.npy, game_test_label.npy існують у {data_path}")
            raise # Перевикидаємо виняток, щоб зупинити програму

        # --- Зміни для StandardScaler ---
        # Reshape data to 2D for StandardScaler: (total_data_points, features)
        # Assuming train_data_raw.shape is (num_samples, win_size, num_features)
        train_data_2d = train_data_raw.reshape(-1, train_data_raw.shape[-1])
        test_data_2d = test_data_raw.reshape(-1, test_data_raw.shape[-1])

        # Fit and transform the 2D data
        self.scaler.fit(train_data_2d) # Навчаємо масштабувальник тільки на тренувальних даних
        train_data_scaled_2d = self.scaler.transform(train_data_2d)
        test_data_scaled_2d = self.scaler.transform(test_data_2d)

        # Reshape data back to 3D for the rest of the model: (num_samples, win_size, num_features)
        self.train = train_data_scaled_2d.reshape(train_data_raw.shape)
        self.test = test_data_scaled_2d.reshape(test_data_raw.shape)
        # --- Кінець змін для StandardScaler ---

        # Важливо: використовуємо частину тренувальних даних для валідації, як у SMDSegLoader
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        # Мітки для тестування.
        # Зверніть увагу: Anomaly Transformer часто використовує test_labels для всіх режимів,
        # тому що під час тренування "мітки" аномалій не використовуються, а для валидації
        # та порогу часто потрібні реальні мітки.
        self.test_labels = test_labels_raw


        print(f"GMSSegLoader initialized:")
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("val:", self.val.shape)
        print("test_labels:", self.test_labels.shape)


    def __len__(self):
        # ... (цей метод не змінюється, залишається таким самим) ...
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else: # mode == 'thre' (для порогу)
            # Зазвичай для 'thre' використовують такі ж дані, як для 'test'
            # або окремий валідаційний набір. Залежить від логіки.
            # Зверніть увагу на використання self.win_size замість self.step в знаменнику для 'thre'
            return (self.test.shape[0] - self.win_size) // self.win_size + 1


    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = np.float32(self.train[index:index + self.win_size])
            if x.ndim == 3 and x.shape[1] == self.win_size:
                x = x[:, 0, :]
            return x, np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            x = np.float32(self.val[index:index + self.win_size])
            if x.ndim == 3 and x.shape[1] == self.win_size:
                x = x[:, 0, :]
            return x, np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            x = np.float32(self.test[index:index + self.win_size])
            if x.ndim == 3 and x.shape[1] == self.win_size:
                x = x[:, 0, :]
            return x, np.float32(self.test_labels[index:index + self.win_size])
        else: # mode == 'thre'
            x = np.float32(self.test[
                index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            if x.ndim == 3 and x.shape[1] == self.win_size:
                x = x[:, 0, :]
            return x, np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class InMemoryGMSSegLoader(GMSSegLoader):
    """Version of :class:`GMSSegLoader` that works with in-memory arrays."""

    def __init__(self, train_data, test_data, test_labels, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data_raw = np.asarray(train_data)
        test_data_raw = np.asarray(test_data)
        test_labels_raw = np.asarray(test_labels)

        train_data_2d = train_data_raw.reshape(-1, train_data_raw.shape[-1])
        test_data_2d = test_data_raw.reshape(-1, test_data_raw.shape[-1])

        self.scaler.fit(train_data_2d)
        train_data_scaled_2d = self.scaler.transform(train_data_2d)
        test_data_scaled_2d = self.scaler.transform(test_data_2d)

        self.train = train_data_scaled_2d.reshape(train_data_raw.shape)
        self.test = test_data_scaled_2d.reshape(test_data_raw.shape)

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]
        self.test_labels = test_labels_raw

        # print("GMSSegLoaderInMemory initialized:")
        # print("test:", self.test.shape)
        # print("train:", self.train.shape)
        # print("val:", self.val.shape)
        # print("test_labels:", self.test_labels.shape)



def get_loader_segment(
    data_path,
    batch_size,
    win_size=100,
    step=100,
    mode='train',
    dataset='KDD',
    num_workers=0,
):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'gms'):
        dataset = GMSSegLoader(data_path, win_size, step, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


def get_loader_segment_from_arrays(
    arrays,
    batch_size,
    win_size=100,
    step=100,
    mode='train',
    dataset='gms',
    num_workers=0,
):
    """Create a data loader from in-memory arrays."""
    if dataset != 'gms':
        raise ValueError("Only 'gms' dataset supported for in-memory loading")

    train_data, test_data, test_labels = arrays
    dataset = InMemoryGMSSegLoader(train_data, test_data, test_labels, win_size, step, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader
