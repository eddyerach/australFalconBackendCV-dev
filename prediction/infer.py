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

##evaluacion en dataset test
#torch.save(model, 'aug5_estado1.pth')
#Cargar el dataset
#name = 'preprocess_area_racimos-bayas_0614_sep10_th01.csv'
#name = 'preprocess_area_racimos-bayas_0627_sep10_th01.csv'
#name = 'preprocess_area_racimos-bayas_3_estado_sep12_th01.csv'
name = 'preprocess_area_racimos-bayas_4_estado_sep12_th01.csv'
##name = 'preprocess_solo1_area_racimos-bayas_4_estado_sep12_th01.csv'
conteo = 'todos'
estado = '4'
fecha = 'sep28'


#datasets train y test por estado 
ds_train = {'1': [92, 3, 31, 34, 59, 69, 86, 8, 75, 30, 51, 54, 88, 40, 99, 93, 85, 62, 79, 9, 82, 26, 76, 47, 29, 6, 42, 78, 16, 52, 43, 23, 37, 39, 38, 64, 97, 22, 5, 66, 81, 58, 27, 73, 77, 20, 25, 55, 45, 12, 96, 41, 44, 35, 13, 71, 19, 91, 56, 11, 63, 72, 100, 65, 1, 67, 17, 57, 18, 10],
            '2': [92, 3, 31, 34, 59, 69, 86, 8, 75, 30, 51, 54, 88, 40, 99, 93, 85, 62, 79, 9, 82, 26, 76, 47, 29, 6, 42, 78, 16, 52, 43, 23, 37, 39, 38, 64, 97, 22, 5, 66, 81, 58, 27, 73, 77, 20, 25, 55, 45, 12, 96, 41, 44, 35, 13, 71, 19, 91, 56, 11, 63, 72, 100, 65, 1, 67, 17, 57, 18, 10], 
            '3': [75, 32, 79, 74, 56, 8, 93, 97, 100, 35, 90, 69, 38, 89, 86, 30, 99, 9, 44, 61, 78, 27, 6, 40, 24, 16, 31, 41, 49, 39, 3, 53, 85, 45, 23, 5, 66, 94, 57, 28, 73, 77, 21, 26, 54, 47, 12, 60, 42, 46, 36, 13, 71, 19, 68, 64, 11, 96, 82, 81, 65, 58, 1, 67, 17, 59, 18, 10], 
            '4': [93, 3, 31, 34, 60, 70, 87, 8, 76, 30, 52, 55, 89, 40, 99, 94, 86, 63, 80, 9, 83, 26, 77, 48, 29, 6, 42, 79, 16, 53, 44, 23, 37, 39, 38, 65, 97, 22, 5, 67, 82, 59, 27, 74, 78, 20, 25, 56, 46, 12, 98, 41, 45, 35, 13, 72, 19, 92, 57, 11, 64, 73, 100, 66, 1, 68, 17, 58, 18],
            '5': [93, 3, 31, 34, 60, 70, 87, 8, 76, 30, 52, 55, 89, 40, 99, 94, 86, 63, 80, 9, 83, 26, 77, 48, 29, 6, 42, 79, 16, 53, 44, 23, 37, 39, 38, 65, 97, 22, 5, 67, 82, 59, 27, 74, 78, 20, 25, 56, 46, 12, 98, 41, 45, 35, 13, 72, 19, 92, 57, 11, 64, 73, 100, 66, 1, 68, 17, 58, 18]}

ds_test  = {'1': [87, 50, 24, 36, 53, 28, 2, 83, 74, 14, 70, 49, 90, 61, 48, 15, 21, 7, 94, 32, 60, 4, 68, 84, 98, 89, 46, 95, 33, 80], 
            '2': [87, 50, 24, 36, 53, 28, 2, 83, 74, 14, 70, 49, 90, 61, 48, 15, 21, 7, 94, 32, 60, 4, 68, 84, 98, 89, 46, 95, 33, 80], 
            '3': [88, 52, 25, 37, 55, 29, 2, 84, 76, 14, 72, 51, 91, 63, 50, 15, 22, 7, 95, 33, 62, 4, 70, 87, 92, 48, 98, 34, 83], 
            '4': [10, 88, 51, 24, 36, 54, 28, 2, 84, 75, 14, 71, 50, 91, 62, 49, 15, 21, 7, 95, 32, 61, 4, 69, 85, 90, 47, 96, 33, 81],
            '5': [10, 88, 51, 24, 36, 54, 28, 2, 84, 75, 14, 71, 50, 91, 62, 49, 15, 21, 7, 95, 32, 61, 4, 69, 85, 90, 47, 96, 33, 81]}
nombre_output = estado + '_' + conteo + '_' + fecha # estado1_man1_sep13
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(name)

new_df_X_names = df[["idracimo","nombre"]] 
new_df_X = df[["l_idx_sol"      , "l_idx_no_sol",	
               "l_det_bayas"    ,	 "c_idx_sol",	
               "c_idx_no_sol"   ,	"c_det_bayas",	
               "r_idx_sol"      ,	"r_idx_no_sol",	
               "r_det_bayas"]]
#new_df_Y = df[[conteo]]
new_df_Y = df[['man1', 'man2', 'horacio']]
new_X_list = np.array(new_df_X.values.tolist()).squeeze()
new_Y_list = np.array(new_df_Y.values.tolist())

X = new_X_list
y = new_Y_list
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)

#Cargar el modelo
model = models.modelRic
model.load_state_dict(torch.load('aug4estado4_man2_sep27.pth'))
model.eval()
#model.eval()
#inferir
y_pred = model(X_test)
print(f'y: {y}')
#print(f'y_est: {y_test}')
row_list = []
for idx, (xdet,yp, yr) in enumerate(zip(X, y_pred, y)):
    #determinar dataset: train o test
    #print(f'xdet: {xdet}')
    tipo_ds = 'n/a'
    if idx in ds_train[estado]:
        tipo_ds = 'train'
    elif idx in ds_test[estado]:
        tipo_ds = 'test'
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
                     'man1': yr.tolist()[0],
                     'man2': yr.tolist()[1],
                     'horacio': yr.tolist()[2],
                     'ds': tipo_ds})
    #break

df = pd.DataFrame(row_list)
df.to_csv('infer_Dataset' + nombre_output+ 'test.csv')

#guardar resultados
row_list = []
'''
for xdet,yp, yr in zip(X_test_org, y_pred, y_test):
    row_list.append({'det': xdet[2], 'pred': yp.tolist()[0], 'real': yr.tolist()[0]})

df = pd.DataFrame(row_list)
df.to_csv('aug5_estado1.csv')
'''