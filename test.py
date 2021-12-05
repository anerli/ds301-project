import matplotlib.pyplot as plt

def test(model, X, Y, train_size, scaler, index_map = lambda x: x):
    model.eval()
    predictY = model(X)

    predictY = predictY.data.numpy()
    actualY = Y.data.numpy()

    predictY = scaler.inverse_transform(predictY)
    actualY = scaler.inverse_transform(actualY)

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