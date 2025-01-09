from time_series_nn.data.dataset import load_data, TimeSeriesDataset
from time_series_nn.utils.tickers import get_ticker_from_file
from torch.utils.data import DataLoader
from pprint import pprint

def collect_data(info):

    data_source = info["conf"]["data_source"]
    percentage   = info["conf"]["percentage"]
    print("  Loading data from %s"%data_source)
    # load full input data source
    data = load_data(data_source)
    if data is None:
      ticker = get_ticker_from_file(info)
      print("Warning: Ticker '%s' has no data"%ticker)
      info["objs"].update({"dataset": None})
      return info
    data_size = len(data)
    info["conf"].update({'data_size': data_size})

    # constraint the source to some percentage of it
    limit = int(data_size * percentage)
    data =  data[1:limit]
    print("  Input data source has %d elements and using %d%%(%d) of them"%
          (data_size, int(percentage*100), len(data)))

    # create torch-based data sources
    dataset = TimeSeriesDataset(data, input_size=24, output_size=1)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    info["objs"].update({"dataset": dataset})
    info["objs"].update({"dataloader": dataloader})
    return info