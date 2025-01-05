

def create_model(model, hs):
    model_obj = None
    if model.lower() == "rnn":
        from time_series_nn.models.rnn import RNNModel
        model_obj = RNNModel(input_size=1, hidden_size=hs, output_size=1)
    elif model.lower() == "gru":
        from time_series_nn.models.gru import GRUModel
        model_obj = GRUModel(input_size=1, hidden_size=hs, output_size=1)
    elif model.lower() == "lstm":
        from time_series_nn.models.lstm import LSTMModel
        model_obj = LSTMModel(input_size=1, hidden_size=hs, output_size=1)

    return model_obj

def get_path_file(output_path, model, epoch, percentage, hidden_size, extension = "path"):
    trained_model = output_path + "/" + \
                str(model) + "_" + \
                str(int(epoch)) + "_" + \
                str(percentage)  + "_" + \
                str(hidden_size) + "." + \
                extension
    return trained_model
