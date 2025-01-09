import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint
from time_series_nn.models.configure_models import get_path_file

def evaluate_model(model, dataloader, device):
    model.eval()
    val_loss = 0
    predictions, actuals = [], []
    accum_losses = []
    criterion = nn.MSELoss()

    iterations = 0
    with torch.no_grad():
        for x, y in dataloader:
            iterations += 1
            x, y = x.to(device), y.to(device)
            outputs = model(x.unsqueeze(-1))
            
            # Cross-validation helps identify the optimal 'hidden_size'
            # by splitting the dataset into multiple folds and testing 
            # various configurations.
            val_loss += criterion(outputs, y).item()

            predictions.append(outputs.cpu().numpy())
            actuals.append(y.cpu().numpy())

        print("Loaded %s time series items"%iterations)
        # calculate losses
        accum_losses = val_loss / len(dataloader)

    return predictions, actuals, accum_losses

def validate_model(info):
    val_loss = 0
    predictions, actuals = [], []
    accum_losses = []

    # get information required to validaate model
    model_obj     = info["objs"]["model"]
    dataloader    = info["objs"]["dataloader"]
    device        = info["objs"]['device']
    criterion     = info["objs"]['criterion']

    # for logging only
    model_type    = info["conf"]["model_type"]
    percentage    = info["conf"]["percentage"]
    data_size     = info["conf"]["data_size"]
    epochs        = info["conf"]['epoch']
    hidden_size   = info["conf"]["hidden_size"]

    # set object to evaluate
    print("  Evaluate %s model with %d%% of data(%d): (epochs=%d, hidden_size=%d)"%
        (model_type.upper(), percentage, data_size, epochs, hidden_size))
    model_obj.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model_obj(x.unsqueeze(-1))
            
            # cross-validation helps identify the optimal 'hidden_size'
            # by splitting the dataset into multiple folds and testing 
            # various configurations.
            val_loss += criterion(outputs, y).item()

            predictions.append(outputs.cpu().numpy())
            actuals.append(y.cpu().numpy())

        accum_losses.append(val_loss / len(dataloader))

    # convert predictions and actuals to numpy arrays
    predictions = np.concatenate(predictions).flatten()
    actuals = np.concatenate(actuals).flatten()

    # saving arrays for plotting
    if info["conf"]["save_img"]:
        info["objs"].update({"predictions": predictions})
        info["objs"].update({"actuals": actuals})
    
    # calculate and print evaluation metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    avg_losses = np.mean(accum_losses)

    info.update({
        "errors": {
            "mae": mae,
            "rmse": rmse,
            'avg_losses': avg_losses
        }
    })
    return info

def create_image(info):

    model_type  = info["conf"]["model_type"]
    actuals     = info["objs"]["actuals"]
    predictions = info["objs"]["predictions"]
    epoch       = info["conf"]["epoch"]
    hidden_size = info["conf"]["hidden_size"]
    output_path = info["conf"]["output_path"]
    percentage  = info["conf"]["percentage"]

    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label="Actual Values", marker="o")
    plt.plot(predictions, label="Predicted Values", marker="x")
    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)

    title = "Model Predictions vs Actuals for model %s (epochs=%d, hidden_size=%d)"% \
            (model_type.lower(), epoch, hidden_size)
    plt.title(title)

    # change info configuration to create an image file name
    image_info = info
    image_info["conf"]["file_prefix"] = "comparison"
    image_info["conf"]["extension"] = "png"
    file_path = get_path_file(image_info)
    print("    Saving comparison in "+file_path)

    if not os.path.isdir(output_path):
        print("Warning: Path not found: %s"%output_path)
        return
    
    # create image for comparison
    plt.savefig(file_path)
    