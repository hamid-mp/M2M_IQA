import torch
import torch.nn as nn
import torchvision

from torchvision.models import vgg16, resnet50, inception_v3
from torch.utils.data import DataLoader




class Classification():

    def __init__(self, Data_Loader=None, phase='Train', model_name='vgg'):

        assert isinstance(Data_Loader, DataLoader), 'Data must be a DataLoader object'
        self.data = Data_Loader

        self.model_name = model_name
    
    def Fine_Tune(self):
        if self.model_name in 



