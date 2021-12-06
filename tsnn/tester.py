import matplotlib.pyplot as plt
#from torch.autograd import Variable
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .preprocessor import TimeSeriesPreprocessor
from .backtester import calculate_return



def test(model: torch.nn.Module, tsp: TimeSeriesPreprocessor, plot_results=False):
    model.eval()
    predictY = model(tsp.X)
    actualY = tsp.Y

    # Caculate loss
    criterion = torch.nn.MSELoss()
    loss = criterion(predictY, actualY)
    print('Testing Loss:', loss.item())

    # Convert torch tensors to numpy arrays
    predictY = predictY.data.numpy()
    actualY = actualY.data.numpy()

    # Transform values back to original sizes
    scaler = tsp.scaler
    predictY = scaler.inverse_transform(predictY)
    actualY = scaler.inverse_transform(actualY)

    original_index_adj = tsp.original_index[tsp.window:]
    index_map = lambda i: original_index_adj[i]

    y_pred = predictY.reshape((-1,)).tolist()
    y_actual = actualY.reshape((-1,)).tolist()
    
    #print(y_actual)

    best = calculate_return(y_actual, y_actual) 
    buyhold = y_actual[-1] / y_actual[0]
    ret = calculate_return(y_pred, y_actual)

    print('Best possible return:', calculate_return(y_actual, y_actual))
    print('Return from buying and holding:', y_actual[-1] / y_actual[0])
    print('Return using prediction:', calculate_return(y_pred, y_actual))
    

    if plot_results: plot(predictY, actualY, tsp.train_size, index_map)

    return best, buyhold, ret

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