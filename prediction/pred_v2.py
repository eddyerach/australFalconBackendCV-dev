import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import copy
import models
import ld 

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# Load and preprocess the data
data = ld.load_data()  # Assuming you have a function to load the data
df = pd.DataFrame.from_records(data)

# Select features and target
features = df[["datos_3-1-3"]]
target = df[["man"]]

new_X_list = np.array(features.values.tolist()).squeeze()
new_Y_list = np.array(target.values.tolist())

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(new_X_list, new_Y_list, train_size=0.6, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)


# Define hyperparameters to search
learning_rates = [0.001, 0.0001, 0.00001]
weight_decays = [1e-4, 1e-5, 1e-6]

n_epochs = 2000
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_val_loss = np.inf
best_model = None
best_hyperparameters = None

# Hyperparameter tuning loop
for lr in learning_rates:
    for wd in weight_decays:
        print(f"Trying lr={lr}, weight_decay={wd}")
        model = models.modelA3
        #model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.MSELoss()
        
        # Training loop
        model.train()
        for epoch in range(n_epochs):
            print(f'{epoch}/{n_epochs}',end="\r")
            optimizer.zero_grad()
            y_pred = model(X_train_tensor)
            loss = loss_fn(y_pred, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_tensor)
            val_loss = loss_fn(y_val_pred, y_val_tensor)
            
        # Compare validation loss with best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_hyperparameters = {'lr': lr, 'weight_decay': wd}
        
        print(f"Validation Loss: {val_loss:.4f}")

# Evaluate best model on test set
best_model.eval()
with torch.no_grad():
    y_test_pred = best_model(X_test_tensor)
    test_loss = loss_fn(y_test_pred, y_test_tensor)

print("Best Hyperparameters:", best_hyperparameters)
print("Final Test Loss with Best Model: %.2f" % test_loss.item())