from tester import test
import pickle
import torch

with open('tsp.pkl', 'rb') as f:
    tsp = pickle.load(f)
model = torch.load('model.pkl')

date_index_adj = tsp.original_index[tsp.window:]

mapper = lambda i: date_index_adj[i]

test(model, tsp.X, tsp.Y, tsp.train_size, tsp.scaler, mapper)