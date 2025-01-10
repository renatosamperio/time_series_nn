import torch
import os

from time_series_nn.data.dataset import load_data, TimeSeriesDataset
from time_series_nn.train.evaluate import evaluate_model
from time_series_nn.models.create_model import create_model, get_path_file
from time_series_nn.utils.io_ops import id_name

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

                if not os.path.exists(trained_model):
                    print("  Warning: File '%s' does not exist"%trained_model)
                    continue

                print("  Evaluating %s model with  %d%% of data: (epochs=%d, hidden_size=%d)"%
                    ( model.upper(), percentage, epoch, hs))

                # Step 1: Load the data
                data = load_data(input_file)  # Replace with the path to your dataset
                slice_size = int(len(data)/10)
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
                
                # Step 4: Calculate and print evaluation metrics
                mae = np.mean(np.abs(predictions - actuals))
                rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
                print(f"    Mean Absolute Error (MAE): {mae:.4f}")
                print(f"    Root Mean Squared Error (RMSE): {rmse:.4f}")

                # Step 5: Plot predictions vs actual values
                file_name = "model_comparison_" + \
                            id_name(model, epoch, percentage, hs) + \
                            ".png"
                if save_img:
                    plt.figure(figsize=(10, 6))
                    plt.plot(actuals,  label="Actual Values1", marker="o")
                    plt.plot(predictions, label="Predicted Values", marker="x")
                    plt.legend()
                    plt.title("Model Predictions vs Actuals for "+model.upper() + " model (epochs="+ str(epoch) + ", hidden_size="+str(hs)+" ) "  )
                    plt.xlabel("Time Step")
                    plt.ylabel("Value")
                    plt.grid(True)
                                
                    file_path = output_path + "/" + file_name
                    print("    Saving comparison in: "+file_name)
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
