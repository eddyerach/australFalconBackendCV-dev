import pandas as pd
from csv import DictReader
import random
import csv 

random.seed(5)
'''
Aumenta todas las variables a la vez con +-5%. Esto sera hara 20 + 1 veces para cada fila. 
El input es un dataset divido por test/train basado en id de racimo. Es decir todos los 100 (clr) estan o en train o en test. 
'''

#dataset_name = 'TEST_area_racimos-bayas_0614_ago19_th01.csv'
dataset_name = 'area_racimos-bayas_0614_ago19_th01.csv'
variacion = 0.05
up = 1 + variacion #aug_limite_superior upper
lo = 1 - variacion #aug_limite_inferior lower
aug_cantidad = 20
augmentable_params = ['area_racimo', 'area_bayas', 'area_bayas_ns', 'det_bayas', 'man']
idx = 0
augmented_dataset = []
#cargar csv como lista de diccionarios
with open(dataset_name, 'r') as f:
    dict_reader = DictReader(f)
    list_of_dict = list(dict_reader)
    print(list_of_dict)

def augment(row, idx, new_name, desc_name, lower_limit, upper_limit):
    area_racimo   = float(row['area_racimo']) * random.uniform(lower_limit,upper_limit)
    area_bayas    = float(row['area_bayas']) * random.uniform(lower_limit,upper_limit)
    area_bayas_ns = float(row['area_bayas_ns']) * random.uniform(lower_limit,upper_limit)
    det_bayas = round(float(row['det_bayas']) * random.uniform(lower_limit,upper_limit))
    letra = row['imagen'].split('.')[0][-1]
    #man = round(float(row['man1']) * random.uniform(lower_limit,upper_limit))
    man = float(row['man1'])
         
    aux_dict = {'idx':              idx, 
                'imagen':           new_name, 
                'desc_name':        desc_name,
                'letra':            letra,
                'area_racimo':      area_racimo, 
                'area_bayas':       area_bayas, 
                'area_bayas_ns':    area_bayas_ns, 
                'idx_sol':          area_bayas / area_racimo, 
                'idx_no_sol':       area_bayas_ns / area_racimo, 
                'det_bayas':        det_bayas, 
                'man':              man}
    return aux_dict
#Iterar por cada fila, generando el aumento

for row in list_of_dict:
    idx_racimo = row['imagen'][:-5]
    desc_name = 'racimo-' + str(idx_racimo) + '_aumento-' + str(0)
    letra = row['imagen'].split('.')[0][-1]
    augmented_dataset.append({
                'idx':              idx, 
                'imagen':           row['imagen'], 
                'letra':            letra,
                'desc_name':        desc_name,
                'area_racimo':      row['area_racimo'], 
                'area_bayas':       row['area_bayas'], 
                'area_bayas_ns':    row['area_bayas_ns'], 
                'idx_sol':          row['idx_sol'], 
                'idx_no_sol':       row['idx_no_sol'], 
                'det_bayas':        row['det_bayas'], 
                'man':              row['man1']})
    idx+=1
    for i in range(1,aug_cantidad+1):
        print(f'i: {i}')
        idx_racimo = row['imagen'][:-5]
        #letra_racimo = row['imagen'].split('.')[0][-1]
        #extension_racimo = row['imagen'].split('.')[1]
        new_name = 'aug_' + str(i) + '_' + row['imagen']
        desc_name = 'racimo-' + str(idx_racimo) + '_aumento-' + str(i)
        aux_dict = augment(row, idx, new_name, desc_name, lo, up)
        #aux_dict = {'idx': idx, 'imagen': new_name, 'area_racimo': float(row['area_racimo']) * random.uniform(lo,up), 'area_bayas': '153825', 'area_bayas_ns': '138024.5', 'idx_sol': '0.553391037817303', 'idx_no_sol': '0.496548163817418', 'det_bayas': '70', 'man1': '141', 'man2': '98', 'split': 'train'}
        #print(f'augmen_value: {augmen_value} {param}')
        augmented_dataset.append(aux_dict)
        idx+=1
        #break
    #break

##Guardar el dataset aumentado en un archivo csv
#keys = list(augmented_dataset[0].keys())
keys = ['idx', 'desc_name', 'imagen', 'letra','area_racimo', 'area_bayas', 'area_bayas_ns', 'idx_sol', 'idx_no_sol', 'det_bayas', 'man']
#print(f'keys: {keys}')
with open('AUG_'+dataset_name, 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(augmented_dataset)