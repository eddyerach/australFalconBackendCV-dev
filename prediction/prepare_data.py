import pandas as pd
from csv import DictReader
import random
import csv 

random.seed(5)
'''
Prepara el dataset para la inferencia.  
El input es un dataset divido por test/train basado en id de racimo. Es decir todos los 100 (clr) estan o en train o en test. 
'''

#dataset_name = 'area_racimos-bayas_0614_sep10_th01.csv'
#dataset_name = 'area_racimos-bayas_0627_sep10_th01.csv'
#dataset_name = 'area_racimos-bayas_3_estado_sep12_th01.csv'
dataset_name = 'area_racimos-bayas_4_estado_sep12_th01.csv'

#variacion = 0.10
#up = 1 + variacion #aug_limite_superior upper
#lo = 1 - variacion #aug_limite_inferior lower
#aug_cantidad = 20
#augmentable_params = ['area_racimo', 'area_bayas', 'area_bayas_ns', 'det_bayas', 'man']
idx = 0
augmented_dataset = []
#cargar csv como lista de diccionarios
with open(dataset_name, 'r') as f:
    dict_reader = DictReader(f)
    list_of_dict = list(dict_reader)
    print(list_of_dict)

for row in list_of_dict:
    #print(f"row['imagen']: {row['imagen']}")
    idx_racimo = int(row['imagen'][:-5])
    desc_name = 'racimo-' + str(idx_racimo) + '_aumento-' + str(0)
    letra = row['imagen'].split('.')[0][-1]
    augmented_dataset.append({
                'idx':              idx, 
                'racimo':           idx_racimo,
                'letra':            letra,
                'aumento':          str(0),
                'area_racimo':      row['area_racimo'], 
                'area_bayas':       row['area_bayas'], 
                'area_bayas_ns':    row['area_bayas_ns'], 
                'idx_sol':          row['idx_sol'], 
                'idx_no_sol':       row['idx_no_sol'], 
                'det_bayas':        row['det_bayas'], 
                'man1':              row['man1'],
                'man2':              row['man2'],
                'horacio':              row['horacio'],
                'horacio_man1':     row['horacio_man1']
                })


##Guardar el dataset aumentado en un archivo csv
#keys = list(augmented_dataset[0].keys())
keys = ['idx', 'racimo', 'letra','aumento','area_racimo', 'area_bayas', 'area_bayas_ns', 'idx_sol', 'idx_no_sol', 'det_bayas', 'man1', 'man2', 'horacio']
#print(f'keys: {keys}')


####
#Crear diccionario intermedio
####
dict_aux_linea = {}
for m in augmented_dataset:
    idx_racimo  = m['racimo'] 
    idx_aumento = m['aumento']
    llave       = 'racimo-'+str(idx_racimo) + '_aumento-' + str(idx_aumento)
    letra       = m['letra']
    idx_sol     = m['idx_sol']	
    idx_no_sol	= m['idx_no_sol']
    det_bayas	= m['det_bayas']
    man1         = m['man1']
    man2         = m['man2']
    horacio         = m['horacio']
    horacio_man1    = m['horacio_man1']

    ##verificar si la llave existe:
    if llave in dict_aux_linea:
        #Si existe, agregar el nuevo diccionario de letra
        dict_aux_linea[llave][letra] = [idx_sol, idx_no_sol, det_bayas]
        dict_aux_linea[llave]['man1'] = man1
        dict_aux_linea[llave]['man2'] = man2
        dict_aux_linea[llave]['horacio'] = horacio
        dict_aux_linea[llave]['horacio_man1'] = horacio_man1
    else: 
        #Si no existe, crear registro y agregar el diccionario de letra
        dict_aux_linea[llave] = {}
        dict_aux_linea[llave][letra] = [idx_sol, idx_no_sol, det_bayas]
        dict_aux_linea[llave]['man1'] = man1
        dict_aux_linea[llave]['man2'] = man2
        dict_aux_linea[llave]['horacio'] = horacio
        dict_aux_linea[llave]['horacio_man1'] = horacio_man1


final_dataset = []
permutaciones = ['LCR']

for p in permutaciones:
    for k1,v1 in dict_aux_linea.items():
        k1 = k1 + '_' + p
        #LCR, CLR, RLC
        #LRC, CRL, RCL
        ##id racimo racimo-6_aumento-0
        print(f'k1: {k1}')
        print(f'v1: {v1}')

        id_racimo = int(k1.split('_')[0].split('-')[1])
        ##valores L
        l_idx_sol       = v1[p[0]][0]
        l_idx_no_sol	= v1[p[0]][1]
        l_det_bayas     = v1[p[0]][2]

        ##valores C
        c_idx_sol       = v1[p[1]][0]
        c_idx_no_sol	= v1[p[1]][1]
        c_det_bayas     = v1[p[1]][2]

        ##valores R
        r_idx_sol       = v1[p[2]][0]
        r_idx_no_sol	= v1[p[2]][1]
        r_det_bayas     = v1[p[2]][2]

        aux_dict = {'idracimo' : id_racimo, 'nombre': k1,  
                    'l_idx_sol': float(l_idx_sol),'l_idx_no_sol': float(l_idx_no_sol), 'l_det_bayas': l_det_bayas,
                    'c_idx_sol': float(c_idx_sol),'c_idx_no_sol': float(c_idx_no_sol), 'c_det_bayas': c_det_bayas,
                    'r_idx_sol': float(r_idx_sol),'r_idx_no_sol': float(r_idx_no_sol), 'r_det_bayas': r_det_bayas,                
                    'man1':v1['man1'],
                    'man2':v1['man2'],
                    'horacio':v1['horacio'],
                    'horacio_man1':v1['horacio_man1'],
                    }
        #print('aux:',aux_dict)
        final_dataset.append(aux_dict)
            #break
        #break
df = pd.DataFrame(final_dataset)
df.to_csv('preprocess_'+dataset_name)

