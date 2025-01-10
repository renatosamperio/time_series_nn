from os import path
import yfinance as yf
import pandas as pd
from optparse import OptionParser
from datetime import datetime
from pprint import pprint

import os, sys

def get_ticker_from_file(info):
    data_source = info["conf"]["data_source"]
    file_prefix = info["conf"]["file_prefix"]
    path_file   = info["conf"]["path_file"]

    start = data_source.find(path_file+"/")+len(path_file) + 1
    end = data_source.find('_'+file_prefix, start)
    ticker = data_source[start:end]
    
    return ticker

def get_stockanalysis_tickers(list_type, path_file):
    '''
        List obtained from https://stockanalysis.com/list
    '''

    path_file = path_file + "/" + list_type + ".csv"
    print("Parsing tickers list for "+path_file)

    if not os.path.exists(path_file):
        print("Error: File "+path_file+" not found")
        return None
    
    return pd.read_csv(path_file)

def download_ticker_data(symbol, start_date, end_date):
    # Create a Ticker object
    stock = yf.Ticker(symbol)

    # Access dividend and stock split history.
    dividends = stock.dividends
    splits = stock.splits

    # Fetch data from the following time
    historical_data = stock.history(start=start_date, end=end_date)

    stock_values = {
        "historical_data": historical_data,
        "dividends": dividends,
        "splits":splits
    }
    return stock_values

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-l", "--list_type", 
        default = None,
        action = "store", 
        type = "string",
        help = "Valid list type as a CSV file")
    parser.add_option("-p", "--path_file", 
        default = None,
        action = "store", 
        type = "string",
        help = "Valid path to CSV file")
    parser.add_option("-s", "--start_date", 
        default = None,
        action = "store", 
        type = "string",
        help = "Input starting date to sample stock values")
    parser.add_option("-e", "--end_date", 
        default = datetime.today().strftime('%Y-%m-%d'), # today's date
        action = "store", 
        type = "string",
        help = "Input ending date to sample stock values")
    parser.add_option("-y", "--symbol", 
        default = None,
        action = "store", 
        type = "string",
        help = "Input ticker symbol")

    (options, args) = parser.parse_args()
    # print(options)

    if not options.list_type:
        parser.error("Required a list type as a CSV file")

    if not options.path_file:
        parser.error("Required a path to input file")

    if not options.start_date:
        parser.error("Required an inital date in format YYYY-MM-DD")

    if not options.symbol:
        parser.error("Required a valid ticker symbol")

    ticker_info = get_stockanalysis_tickers(
            options.list_type, options.path_file)
    
    print("Input CSV file "+options.list_type+" as pandas dataframe")
    print(ticker_info)

    stock_values = download_ticker_data(
        options.symbol, options.start_date,options.end_date)
    
    print("Downloaded data for "+options.symbol+" from "+options.start_date+" to "+options.end_date)
    print(stock_values)
