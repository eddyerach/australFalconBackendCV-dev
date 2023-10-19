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

#name = 'AUG5_area_racimos-bayas_0614_sep10_th01.csv'
#name = 'AUG5_area_racimos-bayas_0627_sep10_th01.csv'
#name = 'AUG5_area_racimos-bayas_3_estado_sep12_th01.csv'
name = 'AUG5_area_racimos-bayas_4_estado_sep12_th01.csv'
#name = 'aug5_solo1_area_racimos-bayas_4_estado_sep12_th01.csv'
estado = 'estado4'
conteo = 'man2'
fecha = 'sep27'
nombre_output = estado + '_' + conteo + '_' + fecha # estado1_man1_sep13
df = pd.read_csv(name)

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
new_df_X_names = df[["idracimo","nombre"]] 
new_df_X = df[["l_idx_sol"      , "l_idx_no_sol",	
               "l_det_bayas"    ,	 "c_idx_sol",	
               "c_idx_no_sol"   ,	"c_det_bayas",	
               "r_idx_sol"      ,	"r_idx_no_sol",	
               "r_det_bayas"]]

#new_df_X = df[["l_det_bayas", "c_det_bayas", "r_det_bayas"]]
#new_df_X_names = df[["nombre"]] 
#new_df_X = df[["datos_3-1-3"]]
#new_df_X = df[["area_racimo", 
#               "area_bayas_ns",
#               "area_bayas",
#               "idx_sol", 
#               "idx_no_sol", 
#               "det_bayas"]]

#new_df_Y = df[["horacio_man1"]]
new_df_Y = df[[conteo]]
#print(f'new_df_X.head(): {new_df_X.head()}')

new_X_list = np.array(new_df_X.values.tolist()).squeeze()
new_Y_list = np.array(new_df_Y.values.tolist())
#print('******************************************************')
#print(new_X_list.shape, type(new_X_list), len(new_X_list), len(new_X_list[0]))
#print(new_Y_list.shape, type(new_Y_list), len(new_Y_list), len(new_Y_list[0]))
#print('******************************************************')
#print(f'new_X_list[0]: {new_X_list[0]}')
#print(f'new_X_list[1]: {new_X_list[1]}')
#print('******************************************************')


X = new_X_list
y = new_Y_list
############# Read data
#data = fetch_california_housing()
#X, y = data.data, data.target
print(f'X: {X[0]}, y:{y[0]}')
# train-test split for model evaluation
#X_train, X_test, y_train, y_test = split_data.split(X, y, train_size=0.7)

#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

##Reemplazar funcion train_test_split, para que el split se haga a nivel de id de racimo, no de muestra
#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.7, shuffle=True)
#X_train, X_test, y_train, y_test = split.train_test_split_racimo(new_df_X_names, X_scaled, y, train_size=0.7, shuffle=True)
X_train, X_test, y_train, y_test = split.train_test_split_racimo(new_df_X_names, X, y, train_size=0.7, shuffle=True)

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