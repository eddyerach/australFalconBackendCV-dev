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
import ld_separados as ld
import models

data_train = ld.load_data('AUG_TRAIN_area_racimos-bayas_0614_ago19_th01.csv')
data_test =  ld.load_data('AUG_TEST_area_racimos-bayas_0614_ago19_th01.csv')
#print(f'data: {data}')

df_train = pd.DataFrame.from_records(data_train)
df_test  = pd.DataFrame.from_records(data_test)
#print(df)


# Assuming you have prepared your X_train and y_train datasets
#df = pd.read_csv('aug_dataset_ago9.csv')
#,imagen,area_racimo,area_bayas,area_bayas_ns,idx_sol,idx_no_sol,det_bayas,man
#new_df_X = df[["area_racimo", "area_bayas", "ab/ar", "det_bayas"]]
new_df_X_train = df_train[["datos_3-1-3"]]
new_df_X_test  = df_test[["datos_3-1-3"]]
#new_df_X = df[["area_racimo", 
#               "area_bayas_ns",
#               "area_bayas",
#               "idx_sol", 
#               "idx_no_sol", 
#               "det_bayas"]]

new_df_Y_train = df_train[["man"]]
new_df_Y_test = df_test[["man"]]
#print(f'new_df_X.head(): {new_df_X.head()}')
#new_df_X.head()
new_X_list_train = np.array(new_df_X_train.values.tolist()).squeeze()
new_X_list_test  = np.array(new_df_X_test.values.tolist()).squeeze()
new_Y_list_train = np.array(new_df_Y_train.values.tolist())
new_Y_list_test  = np.array(new_df_Y_test.values.tolist())
print('******************************************************')
print(new_X_list_train.shape, type(new_X_list_train), len(new_X_list_train), len(new_X_list_train[0]))
print(new_Y_list_train.shape, type(new_Y_list_train), len(new_Y_list_train), len(new_Y_list_train[0]))
print(new_X_list_test.shape, type(new_X_list_test), len(new_X_list_test), len(new_X_list_test[0]))
print(new_Y_list_test.shape, type(new_Y_list_test), len(new_Y_list_test), len(new_Y_list_test[0]))
print('******************************************************')
print(f'new_X_list_train[0]: {new_X_list_train[0]}')
print(f'new_X_list_train[1]: {new_X_list_train[1]}')
print(f'new_X_list_test[0]: {new_X_list_test[0]}')
print(f'new_X_list_test[1]: {new_X_list_test[1]}')
print('******************************************************')


X_train = new_X_list_train
X_test  = new_X_list_test
y_train = new_Y_list_train
y_test  = new_Y_list_test
############# Read data
#data = fetch_california_housing()
#X, y = data.data, data.target
print(f'X: {X_train[0]}, y:{y_train[0]}')
# train-test split for model evaluation
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.7, shuffle=True)

X_test_org = X_test.copy()
# Normalize the input data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
#X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Define the model
model = models.modelRic

# Move data to the selected device
print(f'X_train[0]: {X_train[0]}')
print(f'y_train[0]: {y_train[0]}')
print(f'X_test[0]: {X_test[0]}')
print(f'y_test[0]: {y_test[0]}')
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

n_epochs = 2000   # number of epochs to run
#n_epochs = 1   # number of epochs to run
batch_size = 32  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)


# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []
history_train = []
for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            #move to gpu
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
        print(f'training: {epoch}/{n_epochs}: train loss loss {loss}')
    history_train.append(float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    print(f'Test loss: {mse}')
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
print(f'test_loss len: {len(history)} {history[0]}')
plt.plot(history, label ='test loss')
plt.plot(history_train, label ='train loss')
#plt.ylim(0,600)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title('Train and test loss')
#plt.show()
plt.savefig('l2reg_epochs2000_lr001_arq-modelRic_AUG_0614_loss_1_ago23.png')

##evaluacion en dataset test
torch.save(model, 'l2reg_epochs2000_lr001_arq-modelRic_AUG_0614_model1_ago23.pth')
model.eval()
y_pred = model(X_test)
row_list = []
for xdet,yp, yr in zip(X_test_org, y_pred, y_test):
    row_list.append({'det': xdet[2], 'pred': yp.tolist()[0], 'real': yr.tolist()[0]})


df = pd.DataFrame(row_list)
df.to_csv('l2reg_epochs2000_lr001_arq-modelRic_AUG_dnn2_eval2_0614_ago23_dssep_aug21.csv')