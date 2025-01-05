import torch
import os
import json

# from time_series_nn.data.dataset import load_data, TimeSeriesDataset
# from time_series_nn.train.evaluate import evaluate_model
# from time_series_nn.train.do_evaluate import best_combination
from time_series_nn.train.do_chunk import do_chunk 
from torch.utils.data import DataLoader


from optparse import OptionParser
from pprint import pprint

def choose(options):
    # pprint(options)
    
    # if "chunk" in options.operation:
    print("Splitting input data by percentage...")
    chunked_data = do_chunk(options.input_file, options.chunk)
    
    if "train" in options.operation:
        from time_series_nn.train.do_train import do_train
        print("Training models...")
        do_train(chunked_data, options.output_path, 
                options.model_type, options.epochs_sizes, options.hidden_sizes,
                learning_rate = 0.001, percentage = options.chunk)

    if "evaluate" in options.operation:
        from time_series_nn.train.do_evaluate import do_evaluate, best_combination
        print("Evaluating models...")
        errors = do_evaluate(options.input_file, options.output_path, 
                    options.model_type, options.epochs_sizes, options.hidden_sizes,
                    options.save_img, options.chunk)

        if (len(errors)>3):
            # sort the models performance
            sorted_errors = best_combination(errors, "avg_losses")
            print("Best performers based on 'avg_losses'")
            pprint(sorted_errors[0:3])

            sorted_errors = best_combination(errors, "abs_error")
            print("Best performers based on 'abs_error'")
            pprint(sorted_errors[0:3])

            sorted_errors = best_combination(errors, "sqe_error")
            print("Best performers based on 'sqe_error'")
            pprint(sorted_errors[0:3])

if __name__ == "__main__":
    valid_models = ['gru', 'rnn', 'lstm']
    parser = OptionParser()
    parser.add_option("-o", "--output_path", 
                      default = None,
                      action = "store", 
                      type = "string",
                      help = "Output directory for training files")

    parser.add_option("-e", "--epochs_sizes",
                      default = None,
                      action = "append",
                      type = "int",
                      help = "Input amount of epochs for training")
    
    parser.add_option("-i", "--input_file", 
                      default = None,
                      action = "store", 
                      type = "str",
                      help = "Input time serie in a CSV format (timestamp, value)")

    parser.add_option('-s', "--save_img", 
                      action='store_true',
                      default=False,
                      help = "Input to save plot of comparisons")

    parser.add_option("-p", "--operation",
                      default=[], 
                      nargs=1,
                      choices=('train', 'evaluate', 'chunk'),
                      action='append',
                      help = "Input amount of epochs for training")

    parser.add_option("-m", "--model_type",
                      nargs=1,
                      default = [],
                      choices=('gru', 'rnn', 'lstm'),
                      action = "append",
                      help = "Input type of model (rnn, lstm or gru)")
    
    parser.add_option("-c", "--chunk", 
                      default = 1.0,
                      action = "store", 
                      type = "float",
                      help = "Percentage of data input to use for training")
    
    parser.add_option("-d", "--hidden_sizes", 
                      default = [],
                      action='append',
                      type = "int",
                      help = "Percentage of data input to use for training")

    (options, args) = parser.parse_args()

    if options.input_file is None or not os.path.exists(options.input_file):
        parser.error("Required a existing input file")
    if options.output_path is None or not os.path.exists(options.output_path):
        parser.error("Required an output path")
    if options.epochs_sizes is None:
        parser.error("Required amount of epochs")
    else: 
        reduced_epochs = list(set(options.epochs_sizes))
        reduced_epochs.sort()
        options.epochs_sizes = reduced_epochs
    if not options.model_type:
        parser.error("Invalid models should be of [rnn, lstm, gru]")
    else: 
        reduced_model_list = list(set(options.model_type))
        invalid_elem = list(set(reduced_model_list) - set(valid_models))
        options.model_type = reduced_model_list
        if len(invalid_elem)>0:
            parser.error("Invalid models "+str(invalid_elem)+" should be of type [rnn, lstm, gru]")

    if not options.hidden_sizes:
        parser.error("Invalid hidden size(s)")
    # print(options)
    choose(options)