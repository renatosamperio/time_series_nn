import torch
import os

from time_series_nn.data.dataset import TimeSeriesDataset
from time_series_nn.train.train import train_model
from time_series_nn.models.create_model import create_model, get_path_file
from torch.utils.data import DataLoader

def do_train(data, output_path, model_type, epochs, hidden_sizes, learning_rate=0.001, percentage=1.0):
    data_size = len(data)
    percentage = int(percentage*100)
    # data = load_data(input_file)

    dataset = TimeSeriesDataset(data, input_size=24, output_size=1)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for hs in hidden_sizes:
        for model in model_type:
            model_obj = create_model(model, hs)
            
            # print(model)
            for epoch in epochs:
                train_model_file = get_path_file(output_path, model, epoch, percentage, hs)
                # print(train_model_file)
                
                print("  Training %s model with  %d%% of data(%d): (epochs=%d, hidden_size=%d)"%
                    (model.upper(), percentage, data_size, epoch, hs))
                train_model(model_obj, dataloader, epoch, 
                            learning_rate=learning_rate, 
                            device=device, 
                            save_path=train_model_file)
    