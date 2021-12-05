import matplotlib.pyplot as plt
#from torch.autograd import Variable
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .preprocessor import TimeSeriesPreprocessor



def test(model: torch.nn.Module, tsp: TimeSeriesPreprocessor):
    model.eval()
    predictY = model(tsp.X)
    actualY = tsp.Y

    # Convert torch tensors to numpy arrays
    predictY = predictY.data.numpy()
    actualY = actualY.data.numpy()

    # Transform values back to original sizes
    scaler = tsp.scaler
    predictY = scaler.inverse_transform(predictY)
    actualY = scaler.inverse_transform(actualY)

    original_index_adj = tsp.original_index[tsp.window:]
    index_map = lambda i: original_index_adj[i]

    plot(predictY, actualY, tsp.train_size, index_map)


def test_old(model: torch.nn.Module, X: torch.autograd.Variable, Y: torch.autograd.Variable, train_size: int, scaler: MinMaxScaler, index_map = lambda x: x):
    model.eval()
    predictY = model(X)
    actualY = Y

    # Convert torch tensors to numpy arrays
    predictY = predictY.data.numpy()
    actualY = actualY.data.numpy()

    # Transform values back to original sizes
    predictY = scaler.inverse_transform(predictY)
    actualY = scaler.inverse_transform(actualY)

    plot(predictY, actualY, train_size, index_map)

    # predictY = predictY.data.numpy()
    # actualY = Y.data.numpy()

    # predictY = scaler.inverse_transform(predictY)
    # actualY = scaler.inverse_transform(actualY)

    # plot_indices = list(range(len(actualY)))
    # plot_indices = [index_map(idx) for idx in plot_indices]

    # plt.figure(figsize=(16,9))
    # plt.axvline(x=index_map(train_size), c='r', linestyle='--')

    # plt.ylabel('Close Price ($)')
    # plt.xlabel('Day')
    # plt.plot(plot_indices, actualY, c='blue', label='Actual')
    # plt.plot(plot_indices, predictY, c='magenta', label='Prediction', linestyle='-')
    # plt.legend()
    # plt.show()


'''
index_map: mapping from integers 0,1,2,... to some other index (e.g. a date) for use in plotting 
'''
def plot(predictY: np.array, actualY: np.array, train_size, index_map = lambda x: x):
    plot_indices = list(range(len(actualY)))
    plot_indices = [index_map(idx) for idx in plot_indices]

    plt.figure(figsize=(16,9))
    plt.axvline(x=index_map(train_size), c='r', linestyle='--')

    plt.ylabel('Close Price ($)')
    plt.xlabel('Day')
    plt.plot(plot_indices, actualY, c='blue', label='Actual')
    plt.plot(plot_indices, predictY, c='magenta', label='Prediction', linestyle='-')
    plt.legend()
    plt.show()