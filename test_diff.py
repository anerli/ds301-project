from atc_toolbox.test_suite.accessor import get_df
from tsnn.preprocessor import TimeSeriesPreprocessor
from tsnn.models import LSTM
from tsnn.trainer import train
from tsnn.tester import test

msft = get_df('MSFT')

# X, Y = preprocess(msft, 4)
window=4
test_ratio=0.2
tsp = TimeSeriesPreprocessor(window=window, test_ratio=test_ratio)

tsp.process(msft)

lstm = LSTM(num_classes=1, input_size=1, hidden_size=2, num_layers=1)

train(lstm, tsp.trainX, tsp.trainY, num_epochs=500, learning_rate=0.01)

test(lstm, tsp)


# Test LSTM on completely new data

amd = get_df('AMD')
tsp2 = TimeSeriesPreprocessor(window=window, test_ratio=1)
tsp2.process(amd)

test(lstm, tsp2)