import csv
import random

'''
Script separa un dataset (csv) en test y train basado en el id de racimo. Es decir, todos los racimos 100 (l,c,r) estan o en test o en train.
Input: 
csv generado por el script de inferencia detectron
train size: 70%
Output: 
    csv test
    csv test
'''
#csv_input = 'area_racimos-bayas_0627_ago19_th01.csv'
csv_input = 'area_racimos-bayas_0614_ago19_th01.csv'
train_size = 0.7
#Leer csv como lista de diccionarios
#Cargar datset original
with open(csv_input, 'r') as file:
    reader = csv.DictReader(
        file)
    data = list(reader)
    #print(data)

#Como los indices de van de 1 a 100 se usa esto para generar una lista random, y seleccionar asi los datasets.
lista_racimos = [x for x in range(1,101)]
#print(f'lista_racimos: {lista_racimos}')
#barajar lista
random.seed(5)
random.shuffle(lista_racimos)
#print(f'lista_racimos: {lista_racimos}')

##Seleccionar los indices pertenecientes a cada dataset 
division = int(len(lista_racimos) * train_size)
idx_train = lista_racimos[:division]
idx_test = lista_racimos[division:]
#print(f'idx_train: {idx_train}, {len(idx_train)}')
#print(f'idx_test: {idx_test}, {len(idx_test)}')


#Generar nuevas listas de diccionarios filtrando por train y test
data_train = []
data_test = []
for row in data:
    corrected_name = row['imagen'].replace("..",".")
    row['imagen'] = corrected_name
    indice_racimo = int(corrected_name[:-5])
    #print(f'indice_racimo: {indice_racimo}, {row["imagen"]}')
    if indice_racimo in idx_train:
        data_train.append(row)
    else:
        data_test.append(row)

#Generar nuevos csv usando los diccionarios de cada caso
keys = data_train[0].keys()

with open('TRAIN_area_racimos-bayas_0614_ago19_th01.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data_train)

with open('TEST_area_racimos-bayas_0614_ago19_th01.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data_test)