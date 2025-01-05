from time_series_nn.data.dataset import load_data, TimeSeriesDataset
from torch.utils.data import DataLoader

def collect_data(info):

    data_source = info["conf"]["data_source"]
    percentage   = info["conf"]["percentage"]

    # load full input data source
    data = load_data(data_source)
    data_size = len(data)
    info["conf"].update({'data_size': data_size})

    # constraint the source to some percentage of it
    limit = int(data_size * percentage) 
    data =  data[1:limit]
    print("  Input data source has %d elements and using %d%%(%d)"%
          (data_size, percentage, len(data)))

    # create torch-based data sources
    dataset = TimeSeriesDataset(data, input_size=24, output_size=1)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    info["objs"].updaate({"dataset": dataset})
    info["objs"].updaate({"dataloader": dataloader})
    return info