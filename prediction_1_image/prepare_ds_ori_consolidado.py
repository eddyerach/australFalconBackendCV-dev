import pandas as pd
from csv import DictReader
import random
import csv 

random.seed(5)
'''
Prepara el dataset para la inferencia.  
El input es un dataset divido por test/train basado en id de racimo. Es decir todos los 100 (clr) estan o en train o en test. 
'''

dataset_name = 'area_racimos-bayas_estados1-2-3-4.csv'

idx = 0
augmented_dataset = []
#cargar csv como lista de diccionarios
with open(dataset_name, 'r') as f:
    dict_reader = DictReader(f)
    list_of_dict = list(dict_reader)
    print(list_of_dict)

for row in list_of_dict:
    #print(f"row['imagen']: {row['imagen']}")
    idx_racimo = int(row['imagen'][3:-5])
    desc_name = 'racimo-' + str(idx_racimo) + '_aumento-' + str(0)
    letra = row['imagen'].split('.')[0][-1]
    augmented_dataset.append({
                'idx':              idx, 
                'racimo':           idx_racimo,
                'idracimo':         idx_racimo,
                'imagen':           row['imagen'],
                'nombre':           desc_name,
                'letra':            letra,
                'aumento':          str(0),
                'area_racimo':      row['area_racimo'], 
                'area_bayas':       row['area_bayas'], 
                'area_bayas_ns':    row['area_bayas_ns'],

                'l_idx_sol':          row['idx_sol'], 
                'l_idx_no_sol':       row['idx_no_sol'], 
                'l_det_bayas':        row['det_bayas'], 

                'c_idx_sol':          row['idx_sol'], 
                'c_idx_no_sol':       row['idx_no_sol'], 
                'c_det_bayas':        row['det_bayas'], 

                'r_idx_sol':          row['idx_sol'], 
                'r_idx_no_sol':       row['idx_no_sol'], 
                'r_det_bayas':        row['det_bayas'], 
                'man1':              row['man1'],
                'man2':              row['man2'],
                'horacio':              row['horacio'],
                'horacio_man1':     row['horacio_man1']
                })

#keys = ['idx', 'racimo', 'letra','aumento','area_racimo', 'area_bayas', 'area_bayas_ns', 'idx_sol', 'idx_no_sol', 'det_bayas', 'man1', 'man2', 'horacio']

df = pd.DataFrame(augmented_dataset)
df.to_csv('preprocess_ori_'+dataset_name)