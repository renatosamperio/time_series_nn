from time_series_nn.data.dataset import load_data

def do_chunk(input_data, percentage):
    data = load_data(input_data)
    data_size = len(data)

    limit = int(data_size * percentage) 
    print("  Provided data has %d elements and using %d"%
          (data_size, limit))
    return data[1:limit]
