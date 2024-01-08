import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class RobotDataset(Dataset):
    def __init__(self, features_path, labels_path, transform = None):
        self.features = torch.from_numpy(np.load(features_path))
        self.labels = torch.from_numpy(np.load(labels_path)) 
        self.transform  = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        data, label = self.features[index], self.labels[index]
        return data, label
    
    
