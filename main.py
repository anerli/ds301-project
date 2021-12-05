from atc_toolbox.test_suite.accessor import get_df
from preprocessor import TimeSeriesPreprocessor
from models import LSTM
from trainer import train
from test import test

msft = get_df('MSFT')

date_index = msft.index
print(msft)


# X, Y = preprocess(msft, 4)
tsp = TimeSeriesPreprocessor(window=4, test_ratio=0.2)

tsp.process(msft)

lstm = LSTM(num_classes=1, input_size=1, hidden_size=2, num_layers=1)

train(lstm, tsp.trainX, tsp.trainY, num_epochs=2000, learning_rate=0.01)

test(lstm, tsp.X, tsp.Y, tsp.train_size, tsp.scaler, date_index)