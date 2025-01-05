import torch
import os

from time_series_nn.data.dataset import load_data, TimeSeriesDataset
from time_series_nn.train.evaluate import evaluate_model
from time_series_nn.models.create_model import create_model, get_path_file

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

def do_evaluate(input_file, output_path, model_type, epochs, hidden_sizes, save_img = False, percentage=1.0):
    errors = []
    percentage = int(percentage*100)


    for model in model_type:
        for hs in hidden_sizes:
            model_obj = create_model(model, hs)

            for epoch in epochs:
                trained_model = get_path_file(output_path, model, epoch, percentage, hs)
                # print(trained_model)

                if not os.path.exists(trained_model):
                    print("  Warning: File '%s' does not exist"%trained_model)
                    continue

                print("  Evaluating %s model with  %d%% of data: (epochs=%d, hidden_size=%d)"%
                    ( model.upper(), percentage, epoch, hs))

                # Step 1: Load the data
                data = load_data(input_file)  # Replace with the path to your dataset
                slice_size = int(len(data)/10)
                # print("*** slice_size: %s"%slice_size)
                test_dataset = TimeSeriesDataset(
                    data, input_size=slice_size, output_size=1) 
                test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                
                # Step 2: Define the model (ensure it matches the one used during training)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # model_obj.load_state_dict(torch.load(trained_model)) 
                model_obj.load_state_dict(torch.load(trained_model, weights_only=False))  # Replace with your trained model path
                model_obj.to(device)

                # Step 3: Evaluate the model
                predictions, actuals, avg_losses = evaluate_model(model_obj, test_dataloader, device)

                # Convert predictions and actuals to numpy arrays
                predictions = np.concatenate(predictions).flatten()
                actuals = np.concatenate(actuals).flatten()
                # print(actuals)
                
                # Step 4: Calculate and print evaluation metrics
                mae = np.mean(np.abs(predictions - actuals))
                rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
                print(f"    Mean Absolute Error (MAE): {mae:.4f}")
                print(f"    Root Mean Squared Error (RMSE): {rmse:.4f}")

                # Step 5: Plot predictions vs actual values
                # print(save_img)
                file_name = "model_comparison_" + model.lower() \
                            + "_" + str(epoch) + \
                            "_" + str(percentage) + \
                            "_" + str(hs) + \
                            ".png"
                if save_img:
                    learned_limit = int(len(actuals)*(percentage/100))
                    actual_size = len(actuals)
                    left_part = actuals[learned_limit+1:actual_size]
                    right_part = actuals[:learned_limit]

                    plt.figure(figsize=(10, 6))
                    plt.plot(left_part,  label="Actual Values1", marker="o")
                    plt.plot(right_part, label="Actual Values2", marker="*")
                    plt.plot(predictions, label="Predicted Values", marker="x")
                    plt.legend()
                    plt.title("Model Predictions vs Actuals for "+model.upper() + " model (epochs="+ str(epoch) + ", hidden_size="+str(hs)+" ) "  )
                    plt.xlabel("Time Step")
                    plt.ylabel("Value")
                    plt.grid(True)
                                
                    file_path = output_path + "/" + file_name
                    print("    Saving comparison in: "+file_name)
                    # print(file_path)
                    plt.savefig(file_path)
                
                # Step 6: Store values by absolute error
                errors.append({
                        'abs_error': mae,
                        'sqe_error': rmse,
                        'model': model.lower(),
                        'epochs': epoch,
                        # Storing fold losses by hidden size
                        'hidden_size': hs,
                        'avg_losses': avg_losses,
                        # saved file info
                        "file_name": file_name
                })

    return errors

def best_combination(input_list, sort_key):
    # Check if the sort_key exists in the dictionaries
    if not all(isinstance(item, dict) and sort_key in item for item in input_list):
        raise ValueError(f"All dictionaries must contain the key '{sort_key}'.")

    # Sort the list of dictionaries based on the sort_key
    sorted_data = sorted(input_list, key=lambda x: x[sort_key])
    return sorted_data

def best_combination_keys(input_list, sort_keys):
    
    # Validate that all dictionaries have the specified sort_keys
    for sort_key in sort_keys:
        if not all(sort_key in d for d in input_list):
            raise ValueError(f"All dictionaries must contain the key '{sort_key}'.")
    
    # Validate that the sort_keys correspond to numerical values
    for sort_key in sort_keys:
        if not all(isinstance(d[sort_key], (int, float)) for d in input_list):
            raise ValueError(f"The key '{sort_key}' must correspond to numerical values in all dictionaries.")

    # Sort the list of dictionaries
    # return sorted(input_list, key=lambda x: x[sort_keys])
    return sorted(input_list, key=lambda x: tuple(x[key] for key in sort_keys))
