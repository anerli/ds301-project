import numpy as np
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesPreprocessor:
    # Put any configuration parameters here
    def __init__(self, window, test_ratio=0.20):
        self.window = window
        self.train_ratio = 1.0 - test_ratio
        self.test_ratio = test_ratio

    def process(self, df):
        df.index = list(range(len(df)))
        data = df['Close']
        data = np.array(data)
        # Reshape so shape is (n_samples, n_features) (just one feature)
        data = data.reshape((-1, 1))
        self.scaler = MinMaxScaler()
        data = self.scaler.fit_transform(data)

        x, y = TimeSeriesPreprocessor.sliding(data, self.window)

        self.train_size = int(len(y) * self.train_ratio)
        self.test_size = len(y) - self.train_size

        self.X = Variable(torch.Tensor(x))
        self.Y = Variable(torch.Tensor(y))

        self.trainX = Variable(torch.Tensor(x[0:self.train_size]))
        self.trainY = Variable(torch.Tensor(y[0:self.train_size]))

        self.testX = Variable(torch.Tensor(x[self.train_size:len(x)]))
        self.testY = Variable(torch.Tensor(y[self.train_size:len(y)]))

    @staticmethod
    def sliding(data, window):
        x, y = [], []
        for i in range(window, len(data)):
            x.append(data[i-window:i])
            y.append(data[i])
        return np.array(x), np.array(y)



# def preprocess(df, window):
#     df.index = list(range(len(df)))
#     data = df['Close']
#     data = np.array(data)
#     # Reshape so shape is (n_samples, n_features) (just one feature)
#     data = data.reshape((-1, 1))
#     sc = MinMaxScaler()
#     data = sc.fit_transform(data)
#     data[:5]

#     x, y = sliding(data, window)

#     training_data = data
#     train_size = int(len(y) * 0.67)
#     test_size = len(y) - train_size

#     dataX = Variable(torch.Tensor(np.array(x)))
#     dataY = Variable(torch.Tensor(np.array(y)))

#     # trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
#     # trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

#     # testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
#     # testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

#     #return trainX, trainY, testX, testY, sc
#     return dataX, dataY, sc