import torch
import pickle
import numpy as np

from torch.utils.data import Dataset

class CylinderDataset(Dataset):
    def __init__(self, index=[0], positions=[[0, 0]], mean=None, std=None):
        """
        圆柱绕流数据集
        
        :param index: 快照索引
        :param positions: 传感器位置(h, w)
        """
        super(CylinderDataset, self).__init__()
        with open('D:/workspace_DJJ/data/cylinder/Cy_Taira.pickle', 'rb') as df:
            self.data = torch.from_numpy(pickle.load(df)).float().permute(0, 3, 1, 2)[index, :, :, :]  # s, c, h, w

        self.mean = mean if mean is not None else torch.mean(self.data)
        self.std = std if std is not None else torch.std(self.data)

        # 标准化数据
        self.data = (self.data - self.mean) / self.std

        # 使用高级索引创建稀疏数据张量
        positions = np.array(positions).T  # 转置以分离x和y坐标
        sparse_data = torch.zeros_like(self.data)
        sparse_data[:, :, positions[0], positions[1]] = self.data[:, :, positions[0], positions[1]]

        self.observe = sparse_data

    def __getitem__(self, index):
        return self.observe[index, :], self.data[index, :]

    def __len__(self):
        return self.data.shape[0]