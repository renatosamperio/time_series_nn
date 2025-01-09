import torch
import torch.nn as nn
from pprint import pprint

def get_path_file(info):

    output_path = info["conf"]['output_path']
    percentage  = int(info["conf"]['percentage']*100)
    model       = info["conf"]['model_type']
    epoch       = info["conf"]['epoch']
    hidden_size = info["conf"]['hidden_size']
    extension   = info["conf"]['extension']
    file_prefix = info["conf"]["file_prefix"]
    ticker      = info["conf"]["ticker"]

    if file_prefix!="":
        file_prefix =  file_prefix + "_"

    trained_model = output_path + "/" + file_prefix + \
                ticker + "_" + \
                str(model) + "_" + \
                str(int(epoch)) + "_" + \
                str(percentage) + "_" + \
                str(hidden_size) + \
                "." + extension
    print("  Model will be stored in file %s"%str(trained_model))
    return trained_model


def create_model(info, trained_model = None):

    hidden_size   = info["conf"]['hidden_size']
    model_type    = info["conf"]['model_type']
    learning_rate = info["conf"]['learning_rate']

    # Define name of path file for laoding trained model in evaluation phase
    # or to save the model during the training phase
    path_trained_model = get_path_file(info)
 
    # Create a model from torch.nn
    if not trained_model:
        if model_type.lower() == "rnn":
            from time_series_nn.models.rnn import RNNModel
            model_obj = RNNModel(input_size=1, hidden_size=hidden_size, output_size=1)
        elif model_type.lower() == "gru":
            from time_series_nn.models.gru import GRUModel
            model_obj = GRUModel(input_size=1, hidden_size=hidden_size, output_size=1)
        elif model_type.lower() == "lstm":
            from time_series_nn.models.lstm import LSTMModel
            model_obj = LSTMModel(input_size=1, hidden_size=hidden_size, output_size=1)
    else:
        # Load an existing mode, (ensure it matches the one used during training)
        model_obj.load_state_dict(torch.load(path_trained_model, weights_only=False)) 
                
    # Define the model (ensure it matches the one used during training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_obj.to(device)

    optimizer = torch.optim.Adam(model_obj.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Update central data structure
    info["objs"].update({
        "model": model_obj,
        "optimizer": optimizer,
        "criterion": criterion,
        "device": device
        
    })
    info["conf"].update({"path_trained_model": path_trained_model})

    return info
