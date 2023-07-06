import torch
import torch.nn as nn
import torchvision
from Dataset import DogBreed
from torchvision.models import vgg16, resnet50, inception_v3
from torch.utils.data import DataLoader
from configparser import ConfigParser
from pathlib import Path

path = Path.cwd().joinpath('Configs.ini')
configs = ConfigParser(  )
configs.read(path)



#(configs['Training']['Epochs']))


data_path = Path(__file__).parent.resolve() / 'Dataset' / 'DogBreed'


class Classification():

    def __init__(self, Data_Loader=None, phase='Train'):

        #assert isinstance(Data_Loader, DataLoader), 'Data must be a DataLoader object'
        
        self.data = Data_Loader
        
        model_name = configs['Training']['model']

        if model_name == 'vgg16':
            self.model = vgg16(pretrained=False)
            self.model.classifier[-1] = nn.Linear(4096, int(configs['Training']['num_classes']), bias=True)

        elif model_name == 'resnet50': #fc
            self.model = resnet50(pretrained=False)
            self.model.fc  = nn.Linear(2048, int(configs['Training']['num_classes']), bias=True)
        elif model_name == 'inception_v3':
            self.model = inception_v3(pretrained=False)
            self.model.fc = nn.Linear(2048, int(configs['Training']['num_classes']), bias=True)
        else:
            raise ValueError


    def train(self, train_size=0.85, test_size=0.1):

        train_set, test_set, valid_set = DogBreed(data_path).data_path(train=train_size, test=test_size)


        pass
    def Fine_Tune(self):



        epochs = configs['Training']['Epochs']


        for epoch in range(epochs):
            for x, y in self.data:
                pass
        
            

        
if __name__ == '__main__':
    dataflow = Classification()
    print(dataflow.model.fc)



