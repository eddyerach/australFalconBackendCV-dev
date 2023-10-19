import pandas as pd 
import numpy as np 
import csv 
'''
Aumenta cada variable por +5/-5% por separado. Generando 1 *11
'''
#Definir indice de nuevo dataset
idx = 0

#Definir diccionario de aumentos
dict_augments = {
    'a': {'idx': 0, 'imagen': 'aug_a_', 'area_racimo': 1.05, 'area_bayas': 1, 'area_bayas_ns': 1, 'det_bayas': 1, 'man': 1},
    'b': {'idx': 0, 'imagen': 'aug_b_', 'area_racimo': 0.95, 'area_bayas': 1, 'area_bayas_ns': 1, 'det_bayas': 1, 'man': 1},
    'c': {'idx': 0, 'imagen': 'aug_c_', 'area_racimo': 1, 'area_bayas': 1.05, 'area_bayas_ns': 1, 'det_bayas': 1, 'man': 1},
    'd': {'idx': 0, 'imagen': 'aug_d_', 'area_racimo': 1, 'area_bayas': 0.95, 'area_bayas_ns': 1, 'det_bayas': 1, 'man': 1},
    'e': {'idx': 0, 'imagen': 'aug_e_', 'area_racimo': 1, 'area_bayas': 1, 'area_bayas_ns': 1.05, 'det_bayas': 1, 'man': 1},
    'f': {'idx': 0, 'imagen': 'aug_f_', 'area_racimo': 1, 'area_bayas': 1, 'area_bayas_ns': 0.95, 'det_bayas': 1, 'man': 1},
    'g': {'idx': 0, 'imagen': 'aug_g_', 'area_racimo': 1, 'area_bayas': 1, 'area_bayas_ns': 1, 'det_bayas': 1.05, 'man': 1},
    'h': {'idx': 0, 'imagen': 'aug_h_', 'area_racimo': 1, 'area_bayas': 1, 'area_bayas_ns': 1, 'det_bayas': 0.95, 'man': 1},
    'i': {'idx': 0, 'imagen': 'aug_i_', 'area_racimo': 1, 'area_bayas': 1, 'area_bayas_ns': 1, 'det_bayas': 1, 'man': 1.05},
    'j': {'idx': 0, 'imagen': 'aug_j_', 'area_racimo': 1, 'area_bayas': 1, 'area_bayas_ns': 1, 'det_bayas': 1, 'man': 0.95}
} 

#Cargar datset original
with open('area_racimos-bayas_0627_ago19_th01.csv', 'r') as file:
    reader = csv.DictReader(
        file)
    data = list(reader)
    #print(data)

new_data = []
#Generar dataset aumentado
for row in data:
    #print(f'row: {row}')
    dict_aux_orgs = {'idx': idx, 'imagen': row['imagen'], 'area_racimo': float(row['area_racimo']), 
                    'area_bayas': float(row['area_bayas']), 'area_bayas_ns': float(row['area_bayas_ns']), 
                    'idx_sol':	float(row['area_bayas'])/float(row['area_racimo']), 'idx_no_sol': float(row['area_bayas_ns'])/float(row['area_racimo']) ,
                    'det_bayas': float(row['det_bayas']), 'man': float(row['man1'])}
    new_data.append(dict_aux_orgs)
    idx+=1
    for k1, v1 in dict_augments.items():
        #print(f'k1: {k1}, v1: {v1}')
        area_racimo = v1['area_racimo'] * float(row['area_racimo'])
        area_bayas = v1['area_bayas'] * float(row['area_bayas'])
        area_bayas_ns = v1['area_bayas_ns'] * float(row['area_bayas_ns'])
        dict_aux = {'idx': idx, 'imagen': 'aug_'+k1+'_'+row['imagen'], 'area_racimo': area_racimo, 
                    'area_bayas': area_bayas, 'area_bayas_ns': area_bayas_ns, 
                    'idx_sol':	area_bayas/area_racimo, 'idx_no_sol': area_bayas_ns/area_racimo,
                    'det_bayas': round(v1['det_bayas'] * float(row['det_bayas'])), 'man': round(v1['man'] * float(row['man1']))}
        new_data.append(dict_aux)
        idx+=1
    #break

#print(f'new_data: {new_data}')

##generar csv de data aumentada:

keys = new_data[0].keys()

with open('Aug_area_racimos-bayas_0627_ago19_th01.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(new_data)

