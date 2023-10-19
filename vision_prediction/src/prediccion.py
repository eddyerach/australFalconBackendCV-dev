import numpy as np
import torch
import torch
import torch.nn as nn
import copy
from pickle import load

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Predictor:
    def __init__(self, scaler_path, model_path):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.scaler = load(open(scaler_path, 'rb'))
        self.model = modelRic
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    def predict(self,X):
        X_scaled = self.scaler.transform([X])
        X_scaled_flatten = np.array(X_scaled).squeeze()
        X_test = torch.tensor(X_scaled_flatten, dtype=torch.float32).to(self.device)
        y_pred = self.model(X_test).tolist()[0]
        return y_pred

modelRic = nn.Sequential(
    nn.Linear(9, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
).to(device)