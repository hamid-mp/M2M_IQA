import torch
import torch.nn as nn
import torchvision

from torchvision.models import vgg16, resnet50, inception_v3
from torch.utils.data import DataLoader
from configparser import ConfigParser
from pathlib import Path

path = Path.cwd().joinpath('Configs.ini')
configs = ConfigParser(  )
configs.read(path)



#(configs['Training']['Epochs']))




class Classification():

    def __init__(self, Data_Loader=None, phase='Train'):

        assert isinstance(Data_Loader, DataLoader), 'Data must be a DataLoader object'
        self.data = Data_Loader


    
    def Fine_Tune(self):
        
        model_name = configs['Training']['model']

        if model_name == 'vgg16':
            model = vgg16(pretrained=True)

        elif model_name == 'resnet50':
            model = resnet50(pretrained=True)
        
        elif model_name == 'inception_v3':
            model = inception_v3(pretrained=True)

        else:
            raise ValueError

        
            

        
        



