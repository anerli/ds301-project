

# num_epochs = 2000
# learning_rate = 0.01

# input_size = 1
# hidden_size = 2
# num_layers = 1

# num_classes = 1

# lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

import torch


def train(model, trainX, trainY, num_epochs, learning_rate):
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        outputs = model(trainX)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, trainY)
        
        loss.backward()
        
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))