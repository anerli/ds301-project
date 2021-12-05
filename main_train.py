from atc_toolbox.test_suite.accessor import get_df
from preprocessor import TimeSeriesPreprocessor
from models import LSTM
from trainer import train
import pickle
import torch

msft = get_df('MSFT')

# X, Y = preprocess(msft, 4)
tsp = TimeSeriesPreprocessor(window=4, test_ratio=0.2)

tsp.process(msft)

with open('tsp.pkl', 'wb') as f:
    pickle.dump(tsp, f)

lstm = LSTM(num_classes=1, input_size=1, hidden_size=2, num_layers=1)

train(lstm, tsp.trainX, tsp.trainY, num_epochs=500, learning_rate=0.01)

torch.save(lstm, 'model.pkl')

