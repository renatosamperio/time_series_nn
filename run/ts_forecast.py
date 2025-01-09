import os, sys

from time_series_nn.models.configure_models import create_model
from time_series_nn.train.train import model_train
from time_series_nn.train.evaluate import validate_model, create_image
from time_series_nn.data.chunk import collect_data
from time_series_nn.utils.tickers import get_stockanalysis_tickers
from time_series_nn.utils.tickers import download_ticker_data
from time_series_nn.utils.tickers import get_ticker_from_file
from time_series_nn.utils.function_params import reduce_options
from torch.utils.data import DataLoader

from datetime import datetime
from optparse import OptionParser, OptionGroup
from pprint import pprint

def get_input_files(options):
    tickers_info = get_stockanalysis_tickers(
                options.list_type, options.path_file)
    tickers = list(tickers_info["Symbol"])
    if tickers_info is None:
        parser.error("Invalid list of tickers")
    input_data_sources = []
    for idx, symbol in enumerate(tickers):
        file_path_historical = os.path.join(options.path_file, symbol+'_historical.csv')
        input_data_sources.append(file_path_historical)
    return input_data_sources

def download_ticker(options):
    if not os.path.isdir(options.path_file):
        print("Path "+options.path_file+" does not exists")
        return
    
    print("Getting list of tickers for stock analysis")
    # get a list of tickers from input configuration files
    tickers_info = get_stockanalysis_tickers(
                options.list_type, options.path_file)
    # find stock falues for each ticker
    tickers = list(tickers_info["Symbol"])
    for idx, symbol in enumerate(tickers):
        # download data for each company
        print("  ["+str(idx+1)+"] Downaloding stock values for "+symbol)
        stock_values = download_ticker_data(
            symbol, options.start_date,options.end_date)
        
        file_path_historical = os.path.join(options.path_file, symbol+'_historical.csv')
        print("      Saving historical values into "+file_path_historical)
        stock_values["historical_data"].to_csv(file_path_historical)
        
        file_path_dividends = os.path.join(options.path_file, symbol+'_dividends.csv')
        print("      Saving dividends values into "+file_path_dividends)
        stock_values["dividends"].to_csv(file_path_dividends)

def run_nn(options):
    print("Running neural network")
    # flat down input configuration 
    conf = {
        # control options
        "save_img": options["save_img"],
        "train": options["train"],
        "validate": options["validate"], 
        # user configuration options
        "input_data_sources": options["input_data_sources"],
        "model_type": options["model_type"],
        "hidden_sizes": options["hidden_sizes"],
        "epochs": options["epochs_sizes"],
        "file_prefix": options["file_prefix"],
        "learning_rate": options["learning_rate"] # 0.001
    }

    info = {
        "conf": {},
        "objs": {}
    }

    # Re-locate input configuration into runtime information. These
    # mappings are used in further phases (training or evaluation)
    info["conf"].update({'output_path':   options["output_path"]})
    info["conf"].update({'learning_rate': options["learning_rate"]})
    info["conf"].update({'file_prefix':   options["file_prefix"]})
    info["conf"].update({'percentage':    options["percentage"]})
    info["conf"].update({'save_img':      options["save_img"]})
    info["conf"].update({'train':         options["train"]})
    info["conf"].update({'validate':      options["validate"]})
    info["conf"].update({'path_file':     options["path_file"]})
    info["conf"].update({'extension':     "path"})

    # train with multiple parameters and different data sources
    for data_source in conf["input_data_sources"]:
        info["conf"].update({"data_source": data_source})
        info = collect_data(info)

        ## Adding derived information
        ticker = get_ticker_from_file(info)
        info["conf"].update({'ticker': ticker})

        if info["conf"]["data_size"] is None:
            print("  Avoiding NN model for %s"%data_source)
            continue

        for nn_model in conf["model_type"]:
            info["conf"].update({"model_type": nn_model})
            
            for hs in conf["hidden_sizes"]:
                info["conf"].update({"hidden_size": hs})

                for epoch in conf["epochs"]:
                    info["conf"].update({"epoch": epoch})

                    # Define model for training
                    info = create_model(info)

                    # if it is required to train the model...
                    if info["conf"]["train"]:
                        model_train(info)

                    # if it is required to validate de model...
                    if info["conf"]["validate"]:
                        info = validate_model(info)

                        # if it is required to create an image,
                        # it requies model to be already validated
                        if info["conf"]["save_img"]:
                            create_image(info)

if __name__ == "__main__":
    valid_models = ['gru', 'rnn', 'lstm']
    valid_operations = ['train', 'evaluate']
    parser = OptionParser()
    control = OptionGroup(parser, "Control Options")
    control.add_option('-a', "--save_img", 
                    action='store_true',
                    default=False,
                    help = "Set to save image of evaluated models. Requires '--validate' option.")  
    control.add_option('-t', "--train", 
                    action='store_true',
                    default=False,
                    help = "Set to train given models. Requires configuration options.")
    control.add_option('-v', "--validate", 
                    action='store_true',
                    default=False,
                    help = "Set to validate trained models. Requires configuration options.")
    control.add_option('-w', "--download", 
                    action='store_true',
                    default=False,
                    help = "Set to download stock values.")

    io = OptionGroup(parser, "I/O Options")
    io.add_option("-i", "--input_data_sources", 
                    default = None,
                    action = "append", 
                    type = "str",
                    help = "Input file name of time series in a CSV format (timestamp, value)")
    io.add_option("-u", "--output_path", 
                    default = None,
                    action = "store", 
                    type = "string",
                    help = "Output directory for training files")
    io.add_option("-f", "--file_prefix", 
                    default = None,
                    action = "store", 
                    type = "string",
                    help = "File prefix for output files")

    user = OptionGroup(parser, "User Configuration")
    user.add_option("-r", "--learning_rate", 
                    default = 0.0001,
                    action = "store", 
                    type = "float",
                    help = "File prefix for output files")
    user.add_option("-e", "--epochs_sizes",
                    default = None,
                    action = "append",
                    type = "int",
                    help = "Input amount of epochs for training")
    user.add_option("-c", "--percentage", 
                    default = 1.0,
                    action = "store", 
                    type = "float",
                    help = "Percentage of data input to use for training")
    user.add_option("-d", "--hidden_sizes", 
                    default = [],
                    action='append',
                    type = "int",
                    help = "Percentage of data input to use for training for all input sources")
    user.add_option("-m", "--model_type",
                    nargs=1,
                    default = [],
                    choices=('gru', 'rnn', 'lstm'),
                    action = "append",
                    help = "Input type of model (rnn, lstm or gru)")
    
    tickers = OptionGroup(parser, "Configure Tickers lists")
    tickers.add_option("-l", "--list_type", 
                    default = None,
                    action = "store", 
                    type = "string",
                    help = "Valid list type as a CSV file")
    tickers.add_option("-p", "--path_file", 
                    default = 'input_data',
                    action = "store", 
                    type = "string",
                    help = "Valid path to CSV file")
    tickers.add_option("-s", "--start_date", 
        default = None,
        action = "store", 
        type = "string",
        help = "Input starting date to sample stock values")
    tickers.add_option("-n", "--end_date", 
        default = datetime.today().strftime('%Y-%m-%d'), # today's date
        action = "store", 
        type = "string",
        help = "Input ending date to sample stock values")
    
    parser.add_option_group(control)
    parser.add_option_group(io)
    parser.add_option_group(user)
    parser.add_option_group(tickers)
    
    (options, args) = parser.parse_args()
    # print(options)

    if not options.download and options.input_data_sources is None:
        if options.list_type and options.path_file:
            options.input_data_sources = get_input_files(options)
        else:
            parser.error("Required amount of input data sources")
    elif options.input_data_sources is not None: 
        options.input_
        data_sources = reduce_options(options.input_data_sources)

    if options.output_path is None or not os.path.exists(options.output_path):
        parser.error("Required an output path")
    
    if not options.model_type:
        parser.error("Invalid models should be of [rnn, lstm, gru]")
    else: 
        options.model_type = reduce_options(options.model_type, sorted=False)
        invalid_elem = list(set(options.model_type) - set(valid_models))
        if len(invalid_elem)>0:
            parser.error("Invalid models "+str(invalid_elem)+" should be of type [rnn, lstm, gru]")
    if options.epochs_sizes is None:
        parser.error("Required amount of epochs")
    else: 
        options.epochs_sizes = reduce_options(options.epochs_sizes)

    if not options.hidden_sizes:
        parser.error("Invalid hidden size(s)")
    if not options.download and not options.train and not options.validate:
        parser.error("Invalid operations should be set -w, -t or -v ")

    if options.download and not options.start_date:
        parser.error("Required an inital date in format YYYY-MM-DD")

    if options.download:
        download_ticker(options)
    else:
        run_nn(vars(options))