import torch
from pathlib import Path
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Dataset, random_split




class DogBreed():

    def __init__(self, data_path):
        
        self.data_path = data_path

    def load_data(self, transforms=None):
        
        data = datasets.ImageFolder(root=self.data_path,
        transform=transforms,
        target_transform=None)

        return data


    def data_split(self, transforms=None, **kwargs):
        assert sum(kwargs.values()) <= 1, 'Wrong Portion of Images for Train/Test/Validation Data'
        #assert kwargs.keys() in ['train', 'test', 'validation'], "input Must be [train, test, validation]"
        dataset = self.load_data()
        l = len(dataset)
        train_size = int(kwargs['train']*l)
        test_size = int(kwargs['test']*l)
        valid_size = l - train_size - test_size
        train_set, test_set, validation_set = random_split(dataset, [train_size, test_size, valid_size])
        
        return train_set, test_set, validation_set
    


