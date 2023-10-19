import csv
import numpy as np
import pandas as pd
import random

random.seed(5)
estado = 'estado1'
def train_test_split_racimo(new_df_X_names, X_scaled, Y, train_size=0.7, shuffle=True):
    dataset_intermedio = []
    list_idx = list(set([x[0] for x in new_df_X_names.values.tolist()]))
    print(f'new_df_X_names: {list_idx}, len: {len(list_idx)}')
    if shuffle:
        random.shuffle(list_idx)
    #separar indinces test y train
    limite_train = round(len(list_idx)*train_size)
    print(f'indices totales: {len(list_idx)}, train: {limite_train}')
    train_list = [list_idx[:limite_train]][0] 
    test_list  = [list_idx[limite_train:]][0]
    print(f'*train_list: {train_list}') 
    print(f'*test_list: {test_list}') 
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    print('----------------------------------------------------------')
    print(f'train_list: {len(train_list)}, {train_list}')
    print(f'test_list: {len(test_list)}, {test_list}')
    print(f'shuffled new_df_X_names: {list_idx}, len: {len(list_idx)}')
    print(f'new_df_X_names: {type(new_df_X_names.values.tolist())}')
    print(f'X_scaled: {type(X_scaled)}')
    print(f'Y: {type(Y)}')
    print('----------------------------------------------------------')
    
    for idx, nombre in enumerate(new_df_X_names.values.tolist()):
        #print(f'nombre: {nombre[0]}')
        #print(f'nombre[0]: {nombre[0]}, {type(nombre[0])}')
        if int(nombre[0]) in train_list:
            if Y[idx] != 0:
                #print(f'Agregando {nombre[0]} a train ds')
                X_train.append(X_scaled[idx])
                y_train.append(Y[idx])
        else:
            if Y[idx] != 0:
                X_test.append(X_scaled[idx])
                y_test.append(Y[idx])
        #dict_aux = {'racimo':nombre, 'nombre': nombre, 'X': x, 'Y':y}
        #print(f'dict_aux: {dict_aux}')
    #return 0, 0, 0, 0
    train = list(zip(X_train, y_train))
    test = list(zip(X_test, y_test))
    random.shuffle(train)
    random.shuffle(test)
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    print(f'tipos {type(X_test)}')
    print(f'train len: {len(train)}')
    print(f'test  len: {len(test)}')
    return list(X_train), list(X_test), list(y_train), list(y_test)