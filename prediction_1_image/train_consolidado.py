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
import split_consolidado as split
import random
random.seed(5)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#name = 'AUG5_area_racimos-bayas_0614_sep10_th01.csv'
#name = 'AUG5_area_racimos-bayas_0627_sep10_th01.csv'
#name = 'AUG5_area_racimos-bayas_3_estado_sep12_th01.csv'
#name = 'AUG5_area_racimos-bayas_4_estado_sep12_th01.csv'
#name = 'AUG_area_racimos-bayas_0614_sep10_th01.csv'
name = 'AUG_area_racimos-bayas_estados1-2-3-4.csv'
estado = 'estado1234'
conteo = 'man2'
fecha = 'oct4'
nombre_output = estado + '_' + conteo + '_' + fecha # estado1_man1_sep13
df = pd.read_csv(name)


train_list_id = ['e4-76', 'e4-34', 'e3-12', 'e2-17', 'e4-65', 'e2-44', 'e1-94', 'e1-42', 'e2-62', 'e3-30', 'e2-51', 'e2-39', 'e4-68', 'e3-28', 'e2-7', 'e3-44', 'e2-68', 'e4-52', 'e4-10', 'e4-74', 'e4-19', 'e2-23', 'e2-95', 'e2-40', 'e1-67', 'e2-41', 'e3-95', 'e3-35', 'e4-92', 'e3-33', 'e4-57', 'e3-37', 'e3-76', 'e1-25', 'e4-48', 'e2-55', 'e1-71', 'e2-74', 'e4-96', 'e2-11', 'e4-87', 'e3-68', 'e3-27', 'e4-2', 'e2-57', 'e3-96', 'e2-10', 'e2-100', 'e1-93', 'e1-50', 'e2-37', 'e3-67', 'e2-98', 'e4-71', 'e3-47', 'e1-22', 'e3-52', 'e3-88', 'e3-74', 'e1-57', 'e2-94', 'e4-70', 'e2-26', 'e3-85', 'e4-99', 'e1-74', 'e2-53', 'e3-72', 'e1-27', 'e3-94', 'e4-89', 'e2-88', 'e2-1', 'e3-98', 'e4-44', 'e2-84', 'e3-83', 'e3-55', 'e3-89', 'e2-19', 'e2-34', 'e2-54', 'e4-5', 'e3-82', 'e3-7', 'e1-4', 'e1-61', 'e3-63', 'e3-18', 'e2-75', 'e4-59', 'e2-66', 'e3-41', 'e4-67', 'e1-95', 'e1-97', 'e3-23', 'e2-32', 'e3-93', 'e2-90', 'e4-22', 'e2-52', 'e3-58', 'e4-36', 'e2-61', 'e1-70', 'e1-5', 'e1-96', 'e3-57', 'e2-99', 'e1-81', 'e2-13', 'e4-79', 'e1-52', 'e1-11', 'e3-15', 'e3-64', 'e4-91', 'e2-76', 'e1-35', 'e2-91', 'e4-37', 'e2-20', 'e3-79', 'e1-44', 'e2-24', 'e4-49', 'e2-81', 'e4-9', 'e1-68', 'e4-94', 'e3-38', 'e3-78', 'e4-86', 'e3-61', 'e1-65', 'e3-16', 'e2-18', 'e4-4', 'e3-2', 'e4-81', 'e1-85', 'e1-15', 'e3-42', 'e4-12', 'e4-16', 'e4-58', 'e4-77', 'e3-22', 'e2-63', 'e2-5', 'e1-99', 'e1-82', 'e2-28', 'e1-91', 'e1-23', 'e3-51', 'e4-75', 'e2-12', 'e4-13', 'e2-4', 'e4-85', 'e2-38', 'e2-30', 'e1-3', 'e3-91', 'e3-25', 'e1-34', 'e1-80', 'e2-27', 'e1-24', 'e1-58', 'e1-7', 'e1-86', 'e2-9', 'e3-31', 'e3-59', 'e4-17', 'e1-55', 'e2-70', 'e4-27', 'e3-34', 'e1-17', 'e4-62', 'e2-45', 'e3-71', 'e1-20', 'e2-33', 'e4-35', 'e4-78', 'e2-73', 'e2-35', 'e1-66', 'e2-89', 'e1-90', 'e2-60', 'e4-3', 'e1-9', 'e3-46', 'e1-14', 'e2-36', 'e1-73', 'e3-54', 'e2-79', 'e1-53', 'e3-90', 'e1-32', 'e2-59', 'e4-88', 'e4-39', 'e4-69', 'e1-78', 'e3-66', 'e4-40', 'e4-61', 'e3-97', 'e3-56', 'e3-62', 'e1-47', 'e1-38', 'e3-73', 'e4-7', 'e3-14', 'e1-69', 'e4-28', 'e1-87', 'e3-84', 'e4-20', 'e3-24', 'e2-46', 'e4-90', 'e2-48', 'e2-29', 'e2-80', 'e2-43', 'e4-93', 'e3-70', 'e4-63', 'e4-95', 'e4-83', 'e4-42', 'e3-10', 'e4-6', 'e3-21', 'e4-33', 'e4-31', 'e3-36', 'e3-4', 'e3-65', 'e3-50', 'e4-14', 'e3-45', 'e1-100', 'e1-54', 'e1-36', 'e4-97', 'e3-26', 'e3-8', 'e1-51', 'e2-77', 'e4-55', 'e1-92', 'e1-62', 'e2-21', 'e4-23', 'e1-72', 'e4-30', 'e2-71', 'e2-93', 'e4-1', 'e1-13', 'e2-85', 'e2-65', 'e4-11', 'e3-19', 'e4-32', 'e1-31']
test_list_id = ['e4-21', 'e4-47', 'e3-9', 'e1-40', 'e4-66', 'e4-29', 'e4-82', 'e1-30', 'e3-13', 'e2-87', 'e1-29', 'e3-3', 'e1-45', 'e1-46', 'e4-100', 'e4-64', 'e2-67', 'e1-28', 'e2-15', 'e2-47', 'e1-83', 'e2-58', 'e3-6', 'e4-15', 'e3-81', 'e2-49', 'e3-17', 'e4-46', 'e2-50', 'e4-84', 'e2-69', 'e1-48', 'e2-92', 'e1-56', 'e4-73', 'e3-75', 'e1-10', 'e3-11', 'e3-1', 'e2-72', 'e1-18', 'e1-2', 'e4-41', 'e3-92', 'e4-18', 'e1-89', 'e2-3', 'e2-97', 'e1-76', 'e1-79', 'e1-39', 'e1-43', 'e3-100', 'e1-88', 'e4-25', 'e3-87', 'e4-38', 'e3-48', 'e1-6', 'e3-32', 'e4-24', 'e4-72', 'e1-19', 'e1-49', 'e1-21', 'e1-8', 'e2-82', 'e3-69', 'e1-12', 'e3-99', 'e4-45', 'e3-39', 'e1-60', 'e4-8', 'e2-42', 'e2-64', 'e2-83', 'e1-33', 'e1-37', 'e4-80', 'e2-86', 'e1-16', 'e1-98', 'e4-51', 'e3-77', 'e2-25', 'e1-64', 'e3-29', 'e2-56', 'e4-53', 'e2-22', 'e2-14', 'e4-50', 'e1-1', 'e1-59', 'e3-5', 'e4-56', 'e2-78', 'e1-41', 'e4-98', 'e2-2', 'e4-54', 'e2-6', 'e1-26', 'e2-96', 'e1-63', 'e4-26', 'e3-53', 'e2-16', 'e1-75', 'e1-84', 'e3-40', 'e2-31', 'e4-60', 'e3-60', 'e3-49', 'e1-77', 'e3-86', 'e2-8']
#df = pd.DataFrame.from_records(data)
#df.to_csv('dataset_3-1-3.csv', index=False)
#print(df[df['nombre'] == 'aug_20'][''].values)
#print(df[df['nombre'] == 'aug_20']['datos_3-1-3'].values)
#6L 0.689036284429511	0.614985896405237	62
#6R 0.711132953345574	0.591292080036547	63
#6C 0.732020082123838	0.624038783214735	66
## AUmento revisado


# Assuming you have prepared your X_train and y_train datasets
#df = pd.read_csv('aug_dataset_ago9.csv')
#,imagen,area_racimo,area_bayas,area_bayas_ns,idx_sol,idx_no_sol,det_bayas,man
#new_df_X = df[["area_racimo", "area_bayas", "ab/ar", "det_bayas"]]
print(f'new_df_X.head(): {df.head()}')
new_df_X_names = df[["idracimo","nombre", "imagen", "letra"]] 
new_df_X = df[["l_idx_sol"      , "l_idx_no_sol",	
               "l_det_bayas"    ,	 "c_idx_sol",	
               "c_idx_no_sol"   ,	"c_det_bayas",	
               "r_idx_sol"      ,	"r_idx_no_sol",	
               "r_det_bayas"]]


new_df_Y = df[[conteo]]
#print(f'new_df_X.head(): {new_df_X.head()}')

new_X_list = np.array(new_df_X.values.tolist()).squeeze()
new_Y_list = np.array(new_df_Y.values.tolist())



X = new_X_list
y = new_Y_list

print(f'X: {X[0]}, y:{y[0]}')

X_train, X_test, y_train, y_test = split.train_test_split_racimo(train_list_id, test_list_id,new_df_X_names, X, y, train_size=0.7, shuffle=True)

#print('****************************************************')
#print('Resultados split: ')
##print(X_train, X_test, y_train, y_test)
#print(X_train)
#print('****************************************************')
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


# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []
history_train = []
loss = np.inf

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
plt.plot(history_train, label ='train loss')
plt.plot(history, label ='test loss')
plt.ylim(0,6000)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title('Train and test loss')
#plt.show()
plt.savefig('aug4_solo1_' + nombre_output + '.png')

##evaluacion en dataset test
#torch.save(model.state_dict(), 'aug4estado3_man2_sep13.pth')
torch.save(model.state_dict(), 'aug4' + nombre_output + '.pth')
model.eval()
#y_pred = model(X_test)
y_pred = model(X_copy_tensor)
row_list = []
for xdet,yp, yr in zip(X, y_pred, y):
    row_list.append({
                     'idx_comp_sol1': xdet[0],
                     'idx_comp_nosol1': xdet[1],
                     'det1': xdet[2],
                     'idx_comp_sol2': xdet[3],
                     'idx_comp_nosol2': xdet[4],
                     'det2': xdet[5],
                     'idx_comp_sol3': xdet[6],
                     'idx_comp_nosol3': xdet[7],
                     'det3': xdet[8],
                     'pred': yp.tolist()[0], 
                     'real': yr.tolist()[0]})

df = pd.DataFrame(row_list)
df.to_csv('aug4' + nombre_output+ 'todo.csv')


y_pred = model(X_test)
row_list = []
for xdet,yp, yr in zip(X_test_org, y_pred, y_test):
    row_list.append({
                     'idx_comp_sol1': xdet[0].tolist(),
                     'idx_comp_nosol1': xdet[1].tolist(),
                     'det1': xdet[2].tolist(),
                     'idx_comp_sol2': xdet[3].tolist(),
                     'idx_comp_nosol2': xdet[4].tolist(),
                     'det2': xdet[5].tolist(),
                     'idx_comp_sol3': xdet[6].tolist(),
                     'idx_comp_nosol3': xdet[7].tolist(),
                     'det3': xdet[8].tolist(),
                     'pred': yp.tolist()[0], 
                     'real': yr.tolist()[0]})

df = pd.DataFrame(row_list)
df.to_csv('aug4' + nombre_output+ 'test.csv')

'''
prediccion real y 3 detecciones 
otro grafico con 3 indices no solap
otro grafico con 3 indices solap
'''