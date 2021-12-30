import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class AVADataset(Dataset):
    '''
    Конструктор класса принимает датафрейм с путями к спектрограммам, среднее и СКО, вычисленные по
    исследовательским данным
    Возвращает стандратизованную спектрограмму формы (1, 32, 32) и метку ее класса
    '''
    def __init__(self, dataframe: pd.DataFrame, mean: float, std: float):
        self.spec_paths = dataframe['spec_paths']
        self.labels = dataframe['labels']
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return ((np.load(self.spec_paths[idx])-self.mean)/self.std).reshape(-1,32,32), self.labels[idx]