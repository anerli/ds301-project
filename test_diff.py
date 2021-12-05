from atc_toolbox.test_suite.accessor import get_df
from tsnn.preprocessor import TimeSeriesPreprocessor
from tsnn.models import LSTM
from tsnn.trainer import train
from test import test

msft = get_df('MSFT')

date_index = msft.index
print(msft)


# X, Y = preprocess(msft, 4)
tsp = TimeSeriesPreprocessor(window=4, test_ratio=0.2)

tsp.process(msft)

lstm = LSTM(num_classes=1, input_size=1, hidden_size=2, num_layers=1)

train(lstm, tsp.trainX, tsp.trainY, num_epochs=500, learning_rate=0.01)

date_index_adj = date_index[tsp.window:]

mapper = lambda i: date_index_adj[i]

test(lstm, tsp.X, tsp.Y, tsp.train_size, tsp.scaler, mapper)