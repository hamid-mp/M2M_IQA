import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from Dataset import DogBreed
from torchvision.models import vgg16, resnet50, inception_v3
from torch.utils.data import DataLoader
from configparser import ConfigParser
from pathlib import Path
import math
path = Path.cwd().joinpath('Configs.ini')
configs = ConfigParser(  )
configs.read(path)
train_config = configs.items('Training')

#(configs['Training']['Epochs']))


data_path = Path(__file__).parent.resolve() / 'Dataset' / 'DogBreed'

(Path(__file__).parent / 'model').mkdir(exist_ok=True, parents=True)
class Classification():

    def __init__(self, Data_Loader=None, phase='Train'):

        #assert isinstance(Data_Loader, DataLoader), 'Data must be a DataLoader object'
        
        self.data = Data_Loader
        
        model_name = train_config['model']

        if model_name == 'vgg16':
            self.model = vgg16(pretrained=False)
            self.model.classifier[-1] = nn.Linear(4096, int(train_config['num_classes']), bias=True)

        elif model_name == 'resnet50': #fc
            self.model = resnet50(pretrained=False)
            self.model.fc  = nn.Linear(2048, int(train_config['num_classes']), bias=True)

        elif model_name == 'inception_v3':
            self.model = inception_v3(pretrained=False)
            self.model.fc = nn.Linear(2048, int(train_config['num_classes']), bias=True)
        else:
            raise ValueError

    def early_stopping(train_loss, validation_loss, min_delta, tolerance):

        counter = 0
        val = validation_loss.detach().numpy()
        train = train_loss.detach().numpy()
        if (val - train) > min_delta:
            counter +=1
            if counter >= tolerance:
                return True

    def train(self, train_size=0.85, test_size=0.1):
        '''
        we should add following:
        - save/load model during training
        - add a stop critria
        - use dataloader not original dataset format
        '''
        train_set, test_set, valid_set = DogBreed(data_path).data_split(train=train_size, test=test_size)
        
        train_set = DataLoader(train_set, batch_size=train_config['Batch-size'], shuffle=True)
        valid_set = DataLoader(valid_set, batch_size=train_config['Batch-size'], shuffle=False)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(train_config['learning-rate']))

        train_loss_values = []
        valid_loss_values = []
        epoch_count = []
        least_val_loss = math.inf
        for epoch in range(int(train_config['Epochs'])):
            self.model.train()
            train_loss = 0 # calculate training loss for each epoch
            for i, (x, y) in enumerate(train_set):
                x, y = x.to(device), y.to(device)
                y_pred = self.model(x)

                loss = criterion(y_pred, y)

                train_loss += loss # aggregate batch losses

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                validation_loss = 0
                for x_val, y_val in valid_set:

                    x_val, y_val = x_val.to(device), y_val.to(device)
                    
                    y_pred = self.model(x_val)
                    loss = criterion(y_pred, y_val)
                    validation_loss += loss
            if epoch % 5 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(train_loss.detach().numpy())
                valid_loss_values.append(validation_loss.detach().numpy())
                print(f"Epoch: {epoch} | MAE Train Loss: {train_loss} | MAE Test Loss: {validation_loss} ")

            if validation_loss < least_val_loss:
                least_val_loss = validation_loss
                torch.save(self.model.state_dict(), (Path(__file__).parent / 'model' / 'weight.pt').resolve())

            if self.early_stopping(train_loss,
             validation_loss,
             min_delta=float(train_config['early_stop_min_delta']),
             tolerance = int(train_config['early_stop_TOL'])):
                print("We are at epoch:", i)
                break

                

    def test(self, model_path):
        '''
        This method should be used to get probability output of trained models (get probability map of distorted model and pristine model)
        '''
        pass

    def training_curve(self, epoch_count:list, train_loss:list, valid_loss:list):
        # Plot the loss curves
        plt.plot(epoch_count, train_loss, label="Train loss")
        plt.plot(epoch_count, valid_loss, label="Test loss")
        plt.title("Training and test loss curves")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend();
                    

        
if __name__ == '__main__':
    dataflow = Classification()

    dataflow.train()


