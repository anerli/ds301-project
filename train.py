import math
from atc_toolbox.test_suite.accessor import get_df, get_symbols
from tsnn.preprocessor import TimeSeriesPreprocessor
from tsnn.models import LSTM#, RNN
from tsnn.trainer import train
from tsnn.tester import test
import pickle
import torch
from argparse import ArgumentParser
import os

MODEL_DIR = os.path.join(os.path.realpath(os.path.dirname(__name__)), 'saved_models')

# def train_symbol(model, symbol):
#     df = get_df(symbol)

#     # Don't train on the last 20% of any of the data that way we can test on something
#     tsp = TimeSeriesPreprocessor(window=args.window, test_ratio=0.2)
#     tsp.process(df)

#     train(model, tsp.trainX, tsp.trainY, num_epochs=args.epochs, learning_rate=0.01)


'''
Example command to create a new model trained on MSFT:
py train.py -m msft_model -e 1000 -s MSFT -n

Then can go train the same model on something else too!
py train.py -m msft_model -e 1000 -s AMD

py train.py -m full_model -a -n -e 2000
py train.py -m full_model -a -t 
'''
if __name__ == '__main__':
    parser = ArgumentParser(description='Train your Time-Series NN.')
    parser.add_argument('--new', '-n', action='store_true')
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--epochs', '-e', type=int, default=1000)
    parser.add_argument('--symbol', '-s', type=str, help='Stock ticker symbol to train on.', default='MSFT')
    parser.add_argument('--window', '-w', type=int, default=4)
    parser.add_argument('--test', '-t', action='store_true')
    #parser.add_argument('--plot', '-p', action='store_true')
    parser.add_argument('--all', '-a', action='store_true', help='Test model on all data.')
    print(MODEL_DIR)

    args = parser.parse_args()

    #print(args.new)

    #exit()

    model_fname = os.path.join(MODEL_DIR, args.model + '.pkl')


    
    #, features=['Close', 'High'])

    # with open('tsp.pkl', 'wb') as f:
    #     pickle.dump(tsp, f)

    if args.new:
        #model = LSTM(num_classes=1, input_size=1, hidden_size=2, num_layers=1)
        model = LSTM(num_classes=1, input_size=1, hidden_size=20, num_layers=1)
    else:
        model = torch.load(model_fname)

    tsp = TimeSeriesPreprocessor(window=args.window, test_ratio=0.2)

    if args.all:
        symbols = get_symbols()

        if not args.test:
            for i, symbol in enumerate(symbols):
                print(f'=== Training on {symbol} ({i+1}/{len(symbols)}) ===')
                df = get_df(symbol)
                tsp.process(df)
                train(model, tsp.trainX, tsp.trainY, num_epochs=args.epochs, learning_rate=0.01)
                torch.save(model, model_fname)

        avg_best = 0.0
        avg_buyhold = 0.0
        avg_ret = 0.0

        worst_buyhold = math.inf
        worst_buyhold_symbol = None

        for i, symbol in enumerate(symbols):
            print(f'=== Testing on {symbol} ({i+1}/{len(symbols)}) ===')
            df = get_df(symbol)
            tsp.process(df)
            best, buyhold, ret = test(model, tsp, plot_results=False)

            if buyhold < worst_buyhold:
                worst_buyhold = buyhold
                worst_buyhold_symbol = symbol

            avg_best += best
            avg_buyhold += buyhold
            avg_ret += ret

        print('WORST:', worst_buyhold_symbol)

        avg_best /= len(symbols)
        avg_buyhold /= len(symbols)
        avg_ret /= len(symbols)

        print('Average best possible return:', avg_best)
        print('Average buy & hold return:', avg_buyhold)
        print('Average return using model predictions:', avg_ret)

    else:
        df = get_df(args.symbol)
        # Don't train on the last 20% of any of the data that way we can test on something
        #tsp = TimeSeriesPreprocessor(window=args.window, test_ratio=0.2)
        tsp.process(df)

        if not args.test:
            train(model, tsp.trainX, tsp.trainY, num_epochs=args.epochs, learning_rate=0.01)
    
        torch.save(model, model_fname)

        print('=== Testing ===')
        test(model, tsp, plot_results=True)


