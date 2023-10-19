import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator, RegressorMixin
import torch

class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model, loss_fn, optimizer_class, lr=0.001, weight_decay=0, device='cuda:0'):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device

    def fit(self, X, y):
        self.model.to(self.device)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        self.model.train()
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(X_tensor)
            loss = self.loss_fn(y_pred, y_tensor)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_pred = self.model(X_tensor)
            return y_pred.cpu().numpy()


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model01 = nn.Sequential(
    nn.Linear(9, 8192),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(8192, 4096),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(4096, 2048),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 1),
).to(device)

model0 = nn.Sequential(
    nn.Linear(9, 2048),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 1),
).to(device)

model1 = nn.Sequential(
    nn.Linear(9, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 1),
).to(device)

model1_2 = nn.Sequential(
    nn.Linear(9, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
).to(device)

model1_3 = nn.Sequential(
    nn.Linear(9, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 1),
).to(device)

model2 = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
).to(device)

model3 = nn.Sequential(
    nn.Linear(3, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
).to(device)

model4 = nn.Sequential(
    nn.Linear(3, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
).to(device)


model5 = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
).to(device)


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


modelRic1 = nn.Sequential(
    nn.Linear(9, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
).to(device)


modelRic2 = nn.Sequential(
    nn.Linear(9, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
).to(device)


modelRic2_3 = nn.Sequential(
    nn.Linear(3, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
).to(device)

modelRic3 = nn.Sequential(
    nn.Linear(9, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
).to(device)


modelA1 = nn.Sequential(
    nn.Linear(9, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
).to(device)

modelA2 = nn.Sequential(
    nn.Linear(9, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
).to(device)

modelA3 = nn.Sequential(
    nn.Linear(9, 1024),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
).to(device)


model_sep4 = nn.Sequential(
    nn.Linear(9, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1),
).to(device)


model_sep4_3 = nn.Sequential(
    nn.Linear(3, 256),
    nn.ReLU(),
    #nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    #nn.Dropout(0.2),
    nn.Linear(128, 1),
).to(device)