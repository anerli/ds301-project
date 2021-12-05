import matplotlib.pyplot as plt

def test(model, X, Y, train_size, scaler):
    model.eval()
    train_predict = model(X)

    data_predict = train_predict.data.numpy()
    dataY_plot = Y.data.numpy()

    data_predict = scaler.inverse_transform(data_predict)
    dataY_plot = scaler.inverse_transform(dataY_plot)

    plt.figure(figsize=(16,9))
    plt.axvline(x=train_size, c='r', linestyle='--')

    plt.ylabel('Close Price ($)')
    plt.xlabel('Day')
    plt.plot(dataY_plot, c='blue', label='Actual')
    plt.plot(data_predict, c='magenta', label='Prediction', linestyle='-')
    plt.legend()
    plt.show()