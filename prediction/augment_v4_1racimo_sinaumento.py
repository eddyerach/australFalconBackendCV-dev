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
#dataset_name = 'area_racimos-bayas_0614_ago19_th01.csv'
#dataset_name = 'area_racimos-bayas_0627_sep1_th01.csv'
#dataset_name = 'area_racimos-bayas_0614_sep10_th01.csv'
dataset_name = 'area_racimos-bayas_4_estado_sep12_th01.csv'
variacion = 0.10
up = 1 + variacion #aug_limite_superior upper
lo = 1 - variacion #aug_limite_inferior lower
aug_cantidad = 20
#augmentable_params = ['area_racimo', 'area_bayas', 'area_bayas_ns', 'det_bayas', 'man']
idx = 0
augmented_dataset = []
#cargar csv como lista de diccionarios
with open(dataset_name, 'r') as f:
    dict_reader = DictReader(f)
    list_of_dict = list(dict_reader)
    print(list_of_dict)

def augment(row, idx, idx_racimo, idx_aumento, lower_limit, upper_limit):
    area_racimo   = float(row['area_racimo']) * random.uniform(lower_limit,upper_limit)
    area_bayas    = float(row['area_bayas']) * random.uniform(lower_limit,upper_limit)
    area_bayas_ns = float(row['area_bayas_ns']) * random.uniform(lower_limit,upper_limit)
    det_bayas = round(float(row['det_bayas']) * random.uniform(lower_limit,upper_limit))
    letra = row['imagen'].split('.')[0][-1]
    #man = round(float(row['man1']) * random.uniform(lower_limit,upper_limit))
    man1    = float(row['man1']) * random.uniform(lower_limit,upper_limit)
    man2    = float(row['man2']) * random.uniform(lower_limit,upper_limit)
    horacio = float(row['horacio']) * random.uniform(lower_limit,upper_limit)
    horacio_man1 = float(row['horacio_man1']) * random.uniform(lower_limit,upper_limit)
         
    aux_dict = {'idx':              idx, 
                'racimo':           idx_racimo,
                'letra':            letra,
                'aumento':          idx_aumento,
                'area_racimo':      area_racimo, 
                'area_bayas':       area_bayas, 
                'area_bayas_ns':    area_bayas_ns, 
                'idx_sol':          area_bayas / area_racimo, 
                'idx_no_sol':       area_bayas_ns / area_racimo, 
                'det_bayas':        det_bayas, 
                'man1':             man1,
                'man2':             man2,
                'horacio':          horacio,
                'horacio_man1':     horacio_man1
                }
    return aux_dict
#Iterar por cada fila, generando el aumento

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
    #idx+=1
    #for i in range(1,aug_cantidad+1):
    #    #print(f'i: {i}')
    #    idx_racimo = int(row['imagen'][:-5])
    #    #letra_racimo = row['imagen'].split('.')[0][-1]
    #    #extension_racimo = row['imagen'].split('.')[1]
    #    new_name = 'aug_' + str(i) + '_' + row['imagen']
    #    desc_name = 'racimo-' + str(idx_racimo) + '_aumento-' + str(i)
    #    aux_dict = augment(row, idx, idx_racimo, i, lo, up)
    #    #aux_dict = {'idx': idx, 'imagen': new_name, 'area_racimo': float(row['area_racimo']) * random.uniform(lo,up), 'area_bayas': '153825', 'area_bayas_ns': '138024.5', 'idx_sol': '0.553391037817303', 'idx_no_sol': '0.496548163817418', 'det_bayas': '70', 'man1': '141', 'man2': '98', 'split': 'train'}
    #    #print(f'augmen_value: {augmen_value} {param}')
    #    augmented_dataset.append(aux_dict)
    #    idx+=1
    #    #break
    #break

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

df = pd.DataFrame.from_dict(dict_aux_linea)
df.to_csv('consolidado_sep25.csv')

###
#Crear diccionario final
###
#k1 racimo id aumento id
#v1 [l],[c],[r], man
#k2 l
final_dataset = []
#permutaciones = ['LCR', 'LRC', 'CLR', 'CRL', 'RLC', 'RCL']
permutaciones = ['LCR']
side = ['L', 'C' , 'R']
#for s in side:
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
                    'c_idx_sol': float(l_idx_sol),'c_idx_no_sol': float(l_idx_no_sol), 'c_det_bayas': l_det_bayas,
                    'r_idx_sol': float(l_idx_sol),'r_idx_no_sol': float(l_idx_no_sol), 'r_det_bayas': l_det_bayas,                
                    'man1':v1['man1'],
                    'man2':v1['man2'],
                    'horacio':v1['horacio'],
                    'horacio_man1':v1['horacio_man1'],
                    }
        #print('aux:',aux_dict)
        final_dataset.append(aux_dict)

        aux_dict = {'idracimo' : id_racimo, 'nombre': k1,  
                    'l_idx_sol': float(c_idx_sol),'l_idx_no_sol': float(c_idx_no_sol), 'l_det_bayas': c_det_bayas,
                    'c_idx_sol': float(c_idx_sol),'c_idx_no_sol': float(c_idx_no_sol), 'c_det_bayas': c_det_bayas,
                    'r_idx_sol': float(c_idx_sol),'r_idx_no_sol': float(c_idx_no_sol), 'r_det_bayas': c_det_bayas,                
                    'man1':v1['man1'],
                    'man2':v1['man2'],
                    'horacio':v1['horacio'],
                    'horacio_man1':v1['horacio_man1'],
                    }
        #print('aux:',aux_dict)
        final_dataset.append(aux_dict)

        aux_dict = {'idracimo' : id_racimo, 'nombre': k1,  
                    'l_idx_sol': float(r_idx_sol),'l_idx_no_sol': float(r_idx_no_sol), 'l_det_bayas': r_det_bayas,
                    'c_idx_sol': float(r_idx_sol),'c_idx_no_sol': float(r_idx_no_sol), 'c_det_bayas': r_det_bayas,
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
df.to_csv('aug5_solo1_sin_aug'+dataset_name)