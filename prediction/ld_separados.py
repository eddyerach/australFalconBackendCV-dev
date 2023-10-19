import pandas as pd
from csv import DictReader
import numpy as np


def load_data(file_name):
    # open file in read mode
    with open(file_name, 'r') as f:
        dict_reader = DictReader(f)
        list_of_dict = list(dict_reader)


    dataset_1 = []
    dataset_v1 = dict()
    names_prefix_id = set()
    for row in list_of_dict:
        ext_prefijo = ('_').join(row['imagen'].split('_')[:2])
        prefijo =  ext_prefijo if ext_prefijo != row['imagen'] else 'ori_'
        numero = row['imagen'].split('_')[-1].split('.')[0][:-1]
        letra = row['imagen'].split('_')[-1].split('.')[0][-1].upper()
        #print(f'prefijo: {prefijo}')
        #print(f'numero: {numero}')
        #print(f'letra: {letra}')
        dataset_1.append({'n_completo': row['imagen'],
                          'prefijo': prefijo if prefijo else 'ori_',  
                          'nombre': row['imagen'].split('_')[-1],
                          'id_racimo': numero,
                          'letra': letra,
                          'idx_comp_sol': row['idx_sol'],
                          'idx_comp_no_sol': row['idx_no_sol'],
                          'man': row['man']})
        dict_aux =       {'idx_s':       row['idx_sol'],
                          'idx_ns':    row['idx_no_sol'],
                          'det': row['det_bayas']
                            }
        key_dict = prefijo + numero
        if key_dict not in dataset_v1:
            dataset_v1[key_dict] = {}

        dataset_v1[key_dict][letra] = dict_aux
        dataset_v1[key_dict]['man'] = row['man']
        #dataset_v1[prefijo + numero] = dict_aux
        #names_prefix_id.add(('_').join(row['imagen'].split('_')[:2]) + '-' + row['imagen'].split('_')[-1].split('.')[0][:-1])
    #
    #df1 = pd.DataFrame(dataset_1)
    #print(dataset_v1)

    ######## Construir dataset a partir del diccionario
    #formato 3x1x3 
    dataset_v2 = []
    for k,v in dataset_v1.items():
        #print(f'k: {k}, v: {v}')
        #print(f'v[L]: {v["L"]}')
        #print(f'v[L][idx_s]: {v["L"]["idx_s"]}')
        #break
        #dataset_v2.append({'nombre':         k,
        #                   'datos_3-1-3':    np.array([
        #                                                [(v['L']['idx_s'], v['L']['idx_ns'], v['L']['det'])],
        #                                                [(v['C']['idx_s'], v['C']['idx_ns'], v['C']['det'])],
        #                                                [(v['R']['idx_s'], v['R']['idx_ns'], v['R']['det'])]
        #                                    ], dtype = float),  #3d
        #                  #'datos_3-3-1':    np.array([(1.5,2,3), (4,5,6)], dtype = float), #2d
        #                  'man':            float(v['man'])
        #                  })
        #print(f'k: {k}, v: {v}')
        dataset_v2.append({'nombre':         k,
                           'datos_3-1-3':    np.array([v['L']['idx_s'], v['L']['idx_ns'], v['L']['det'],
                                                        v['C']['idx_s'], v['C']['idx_ns'], v['C']['det'],
                                                        v['R']['idx_s'], v['R']['idx_ns'], v['R']['det']
                                            ], dtype = float),  #3d
                          #'datos_3-3-1':    np.array([(1.5,2,3), (4,5,6)], dtype = float), #2d
                          'man':            float(v['man'])
                          })
        
        #print(f'dataset_v2: {dataset_v2}')
        #print(f'shape: {dataset_v2[0]["datos_3-1-3"].shape}')
        #break
    return dataset_v2