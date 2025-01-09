import torch
import time

from pprint import pprint
from torch.utils.data import DataLoader
import torch.nn as nn

def train_model(model, dataloader, epochs, learning_rate, device, save_path):
    start = time.time()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x.unsqueeze(-1))
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    elapsed = time.time() - start
    print(f"    Model saved in {save_path} in {round(elapsed, 2)}s")

def model_train(info):
    
    start = time.time()
    model_obj     = info["objs"]["model"]
    dataloader    = info["objs"]["dataloader"]
    device        = info["objs"]['device']
    criterion     = info["objs"]['criterion']
    optimizer     = info["objs"]['optimizer']
    epochs        = info["conf"]['epoch']
    save_path     = info["conf"]["path_trained_model"]

    # for logging only
    model_type    = info["conf"]["model_type"]
    percentage    = info["conf"]["percentage"]
    data_size     = info["conf"]["data_size"]
    hidden_size   = info["conf"]["hidden_size"]

    print("  Training %s model with %s%% of data(%d): (epochs=%d, hidden_size=%d)"%
        (model_type.upper(), str(int(percentage*100)), data_size, epochs, hidden_size))
    
    # set model for training
    model_obj.train()

    for e in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model_obj(x.unsqueeze(-1))
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(model_obj.state_dict(), save_path)
    elapsed = time.time() - start
    print(f"    Model created in {save_path} in {round(elapsed, 2)}s")