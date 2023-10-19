import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import copy
from sklearn.preprocessing import StandardScaler
import copy
#import ld
import models
import split_data
import split

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
name = 'AUG_area_racimos-bayas_estados1-2-3-4.csv'
estado = 'estado1'
conteo = 'man2'
fecha = 'sep29'
nombre_output = estado + '_' + conteo + '_' + fecha # estado1_man1_sep13
df = pd.read_csv(name)
#print(f'new_df_X.head(): {df.head()}')
new_df_X_names = df[["idracimo","nombre"]] 
new_df_X = df[["l_idx_sol"      , "l_idx_no_sol",	
               "l_det_bayas"    ,	 "c_idx_sol",	
               "c_idx_no_sol"   ,	"c_det_bayas",	
               "r_idx_sol"      ,	"r_idx_no_sol",	
               "r_det_bayas"]]

new_df_Y = df[[conteo]]
new_X_list = np.array(new_df_X.values.tolist()).squeeze()
new_Y_list = np.array(new_df_Y.values.tolist())

X = new_X_list
y = new_Y_list
#print(f'X: {X[0]}, y:{y[0]}')
X_train, X_test, y_train, y_test = split.train_test_split_racimo(new_df_X_names, X, y, train_size=0.7, shuffle=True)

print('****************************************************')
print('Resultados split: ')
print(X_train, X_test, y_train, y_test)
#print(X_train)
print('****************************************************')
X_test_org = X_test.copy()
X_copy = X.copy()
y_copy = y.copy()
# Normalize the input data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_copy_scaled = scaler.transform(X_copy)
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
#X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
#todo el dataset
X_copy_tensor = torch.tensor(X_copy_scaled, dtype=torch.float32).to(device)
y_copy_tensor = torch.tensor(y_copy, dtype=torch.float32).reshape(-1, 1).to(device)



# Define the model
model = models.modelRic

# Move data to the selected device
print(f'X_train[0]: {X_train}')
print(f'y_train[0]: {y_train}')
print(f'X_test[0]: {X_test}')
print(f'y_test[0]: {y_test}')
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)

n_epochs = 1000   # number of epochs to run
#n_epochs = 1   # number of epochs to run
batch_size = 256  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

'''
prediccion real y 3 detecciones 
otro grafico con 3 indices no solap
otro grafico con 3 indices solap
'''