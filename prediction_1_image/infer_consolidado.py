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
from pickle import dump
##evaluacion en dataset test
#torch.save(model, 'aug5_estado1.pth')
#Cargar el dataset
#name = 'preprocess_area_racimos-bayas_0614_sep10_th01.csv'
#name = 'preprocess_area_racimos-bayas_0627_sep10_th01.csv'
#name = 'preprocess_area_racimos-bayas_3_estado_sep12_th01.csv'
#name = 'preprocess_ori_area_racimos-bayas_0614_sep10_th01.csv'
##name = 'preprocess_solo1_area_racimos-bayas_4_estado_sep12_th01.csv'
name = 'preprocess_ori_area_racimos-bayas_estados1-2-3-4.csv'
conteo = 'todos'
estado = 'c'
fecha = 'oct4'


#datasets train y test por estado 
ds_train = {'1': [92, 3, 31, 34, 59, 69, 86, 8, 75, 30, 51, 54, 88, 40, 99, 93, 85, 62, 79, 9, 82, 26, 76, 47, 29, 6, 42, 78, 16, 52, 43, 23, 37, 39, 38, 64, 97, 22, 5, 66, 81, 58, 27, 73, 77, 20, 25, 55, 45, 12, 96, 41, 44, 35, 13, 71, 19, 91, 56, 11, 63, 72, 100, 65, 1, 67, 17, 57, 18, 10],
            '2': [92, 3, 31, 34, 59, 69, 86, 8, 75, 30, 51, 54, 88, 40, 99, 93, 85, 62, 79, 9, 82, 26, 76, 47, 29, 6, 42, 78, 16, 52, 43, 23, 37, 39, 38, 64, 97, 22, 5, 66, 81, 58, 27, 73, 77, 20, 25, 55, 45, 12, 96, 41, 44, 35, 13, 71, 19, 91, 56, 11, 63, 72, 100, 65, 1, 67, 17, 57, 18, 10], 
            '3': [75, 32, 79, 74, 56, 8, 93, 97, 100, 35, 90, 69, 38, 89, 86, 30, 99, 9, 44, 61, 78, 27, 6, 40, 24, 16, 31, 41, 49, 39, 3, 53, 85, 45, 23, 5, 66, 94, 57, 28, 73, 77, 21, 26, 54, 47, 12, 60, 42, 46, 36, 13, 71, 19, 68, 64, 11, 96, 82, 81, 65, 58, 1, 67, 17, 59, 18, 10], 
            '4': [93, 3, 31, 34, 60, 70, 87, 8, 76, 30, 52, 55, 89, 40, 99, 94, 86, 63, 80, 9, 83, 26, 77, 48, 29, 6, 42, 79, 16, 53, 44, 23, 37, 39, 38, 65, 97, 22, 5, 67, 82, 59, 27, 74, 78, 20, 25, 56, 46, 12, 98, 41, 45, 35, 13, 72, 19, 92, 57, 11, 64, 73, 100, 66, 1, 68, 17, 58, 18],
            '5': [93, 3, 31, 34, 60, 70, 87, 8, 76, 30, 52, 55, 89, 40, 99, 94, 86, 63, 80, 9, 83, 26, 77, 48, 29, 6, 42, 79, 16, 53, 44, 23, 37, 39, 38, 65, 97, 22, 5, 67, 82, 59, 27, 74, 78, 20, 25, 56, 46, 12, 98, 41, 45, 35, 13, 72, 19, 92, 57, 11, 64, 73, 100, 66, 1, 68, 17, 58, 18],
            'c': ['e4-76', 'e4-34', 'e3-12', 'e2-17', 'e4-65', 'e2-44', 'e1-94', 'e1-42', 'e2-62', 'e3-30', 'e2-51', 'e2-39', 'e4-68', 'e3-28', 'e2-7', 'e3-44', 'e2-68', 'e4-52', 'e4-10', 'e4-74', 'e4-19', 'e2-23', 'e2-95', 'e2-40', 'e1-67', 'e2-41', 'e3-95', 'e3-35', 'e4-92', 'e3-33', 'e4-57', 'e3-37', 'e3-76', 'e1-25', 'e4-48', 'e2-55', 'e1-71', 'e2-74', 'e4-96', 'e2-11', 'e4-87', 'e3-68', 'e3-27', 'e4-2', 'e2-57', 'e3-96', 'e2-10', 'e2-100', 'e1-93', 'e1-50', 'e2-37', 'e3-67', 'e2-98', 'e4-71', 'e3-47', 'e1-22', 'e3-52', 'e3-88', 'e3-74', 'e1-57', 'e2-94', 'e4-70', 'e2-26', 'e3-85', 'e4-99', 'e1-74', 'e2-53', 'e3-72', 'e1-27', 'e3-94', 'e4-89', 'e2-88', 'e2-1', 'e3-98', 'e4-44', 'e2-84', 'e3-83', 'e3-55', 'e3-89', 'e2-19', 'e2-34', 'e2-54', 'e4-5', 'e3-82', 'e3-7', 'e1-4', 'e1-61', 'e3-63', 'e3-18', 'e2-75', 'e4-59', 'e2-66', 'e3-41', 'e4-67', 'e1-95', 'e1-97', 'e3-23', 'e2-32', 'e3-93', 'e2-90', 'e4-22', 'e2-52', 'e3-58', 'e4-36', 'e2-61', 'e1-70', 'e1-5', 'e1-96', 'e3-57', 'e2-99', 'e1-81', 'e2-13', 'e4-79', 'e1-52', 'e1-11', 'e3-15', 'e3-64', 'e4-91', 'e2-76', 'e1-35', 'e2-91', 'e4-37', 'e2-20', 'e3-79', 'e1-44', 'e2-24', 'e4-49', 'e2-81', 'e4-9', 'e1-68', 'e4-94', 'e3-38', 'e3-78', 'e4-86', 'e3-61', 'e1-65', 'e3-16', 'e2-18', 'e4-4', 'e3-2', 'e4-81', 'e1-85', 'e1-15', 'e3-42', 'e4-12', 'e4-16', 'e4-58', 'e4-77', 'e3-22', 'e2-63', 'e2-5', 'e1-99', 'e1-82', 'e2-28', 'e1-91', 'e1-23', 'e3-51', 'e4-75', 'e2-12', 'e4-13', 'e2-4', 'e4-85', 'e2-38', 'e2-30', 'e1-3', 'e3-91', 'e3-25', 'e1-34', 'e1-80', 'e2-27', 'e1-24', 'e1-58', 'e1-7', 'e1-86', 'e2-9', 'e3-31', 'e3-59', 'e4-17', 'e1-55', 'e2-70', 'e4-27', 'e3-34', 'e1-17', 'e4-62', 'e2-45', 'e3-71', 'e1-20', 'e2-33', 'e4-35', 'e4-78', 'e2-73', 'e2-35', 'e1-66', 'e2-89', 'e1-90', 'e2-60', 'e4-3', 'e1-9', 'e3-46', 'e1-14', 'e2-36', 'e1-73', 'e3-54', 'e2-79', 'e1-53', 'e3-90', 'e1-32', 'e2-59', 'e4-88', 'e4-39', 'e4-69', 'e1-78', 'e3-66', 'e4-40', 'e4-61', 'e3-97', 'e3-56', 'e3-62', 'e1-47', 'e1-38', 'e3-73', 'e4-7', 'e3-14', 'e1-69', 'e4-28', 'e1-87', 'e3-84', 'e4-20', 'e3-24', 'e2-46', 'e4-90', 'e2-48', 'e2-29', 'e2-80', 'e2-43', 'e4-93', 'e3-70', 'e4-63', 'e4-95', 'e4-83', 'e4-42', 'e3-10', 'e4-6', 'e3-21', 'e4-33', 'e4-31', 'e3-36', 'e3-4', 'e3-65', 'e3-50', 'e4-14', 'e3-45', 'e1-100', 'e1-54', 'e1-36', 'e4-97', 'e3-26', 'e3-8', 'e1-51', 'e2-77', 'e4-55', 'e1-92', 'e1-62', 'e2-21', 'e4-23', 'e1-72', 'e4-30', 'e2-71', 'e2-93', 'e4-1', 'e1-13', 'e2-85', 'e2-65', 'e4-11', 'e3-19', 'e4-32', 'e1-31']}

ds_test  = {'1': [87, 50, 24, 36, 53, 28, 2, 83, 74, 14, 70, 49, 90, 61, 48, 15, 21, 7, 94, 32, 60, 4, 68, 84, 98, 89, 46, 95, 33, 80], 
            '2': [87, 50, 24, 36, 53, 28, 2, 83, 74, 14, 70, 49, 90, 61, 48, 15, 21, 7, 94, 32, 60, 4, 68, 84, 98, 89, 46, 95, 33, 80], 
            '3': [88, 52, 25, 37, 55, 29, 2, 84, 76, 14, 72, 51, 91, 63, 50, 15, 22, 7, 95, 33, 62, 4, 70, 87, 92, 48, 98, 34, 83], 
            '4': [10, 88, 51, 24, 36, 54, 28, 2, 84, 75, 14, 71, 50, 91, 62, 49, 15, 21, 7, 95, 32, 61, 4, 69, 85, 90, 47, 96, 33, 81],
            '5': [10, 88, 51, 24, 36, 54, 28, 2, 84, 75, 14, 71, 50, 91, 62, 49, 15, 21, 7, 95, 32, 61, 4, 69, 85, 90, 47, 96, 33, 81],
            'c': ['e4-21', 'e4-47', 'e3-9', 'e1-40', 'e4-66', 'e4-29', 'e4-82', 'e1-30', 'e3-13', 'e2-87', 'e1-29', 'e3-3', 'e1-45', 'e1-46', 'e4-100', 'e4-64', 'e2-67', 'e1-28', 'e2-15', 'e2-47', 'e1-83', 'e2-58', 'e3-6', 'e4-15', 'e3-81', 'e2-49', 'e3-17', 'e4-46', 'e2-50', 'e4-84', 'e2-69', 'e1-48', 'e2-92', 'e1-56', 'e4-73', 'e3-75', 'e1-10', 'e3-11', 'e3-1', 'e2-72', 'e1-18', 'e1-2', 'e4-41', 'e3-92', 'e4-18', 'e1-89', 'e2-3', 'e2-97', 'e1-76', 'e1-79', 'e1-39', 'e1-43', 'e3-100', 'e1-88', 'e4-25', 'e3-87', 'e4-38', 'e3-48', 'e1-6', 'e3-32', 'e4-24', 'e4-72', 'e1-19', 'e1-49', 'e1-21', 'e1-8', 'e2-82', 'e3-69', 'e1-12', 'e3-99', 'e4-45', 'e3-39', 'e1-60', 'e4-8', 'e2-42', 'e2-64', 'e2-83', 'e1-33', 'e1-37', 'e4-80', 'e2-86', 'e1-16', 'e1-98', 'e4-51', 'e3-77', 'e2-25', 'e1-64', 'e3-29', 'e2-56', 'e4-53', 'e2-22', 'e2-14', 'e4-50', 'e1-1', 'e1-59', 'e3-5', 'e4-56', 'e2-78', 'e1-41', 'e4-98', 'e2-2', 'e4-54', 'e2-6', 'e1-26', 'e2-96', 'e1-63', 'e4-26', 'e3-53', 'e2-16', 'e1-75', 'e1-84', 'e3-40', 'e2-31', 'e4-60', 'e3-60', 'e3-49', 'e1-77', 'e3-86', 'e2-8']}
nombre_output = estado + '_' + conteo + '_' + fecha # estado1_man1_sep13
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(name)

new_df_X_names = df[["idracimo","nombre", "imagen"]]
xnames = df[["imagen"]]
new_df_X = df[["l_idx_sol"      , "l_idx_no_sol",	
               "l_det_bayas"    ,	 "c_idx_sol",	
               "c_idx_no_sol"   ,	"c_det_bayas",	
               "r_idx_sol"      ,	"r_idx_no_sol",	
               "r_det_bayas"]]
#new_df_Y = df[[conteo]]
new_df_Y = df[['man1', 'man2', 'horacio']]
new_X_list = np.array(new_df_X.values.tolist()).squeeze()
xnames_list = np.array(xnames.values.tolist()).squeeze()
new_Y_list = np.array(new_df_Y.values.tolist())

X = new_X_list
y = new_Y_list
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
dump(scaler, open('scaler.pkl', 'wb'))
X_test_scaled = scaler.transform(X)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)

#Cargar el modelo
model = models.modelRic
model.load_state_dict(torch.load('aug4estado1234_man2_sep29.pth'))
model.eval()
#model.eval()
#inferir
y_pred = model(X_test)
print(f'y: {y}')
#print(f'y_est: {y_test}')
row_list = []
for idx, (xname, xdet,yp, yr) in enumerate(zip(xnames_list,X, y_pred, y)):
    #determinar dataset: train o test
    xname = xname[:-5]
    print(f'xdet: {xname}')
    tipo_ds = 'n/a'
    print(X)
    if xname in ds_train[estado]:
        tipo_ds = 'train'
    elif xname in ds_test[estado]:
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