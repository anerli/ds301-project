import pickle
import torch
from tsnn.trainer import train

with open('tsp.pkl', 'rb') as f:
    tsp = pickle.load(f)
model = torch.load('model.pkl')

train(model, tsp.trainX, tsp.trainY, num_epochs=500, learning_rate=0.01)

torch.save(model, 'model.pkl')