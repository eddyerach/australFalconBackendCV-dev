def split(X, y, train_size=0.7):
    lista_racimos = [x for x in range(1,101)] # define lista de 1 a 100
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
    for x1, y1  in zip(X,y):
        corrected_name = row['imagen'].replace("..",".")
        row['imagen'] = corrected_name
        indice_racimo = int(corrected_name[:-5])
        #print(f'indice_racimo: {indice_racimo}, {row["imagen"]}')
        if indice_racimo in idx_train:
            data_train.append(row)
        else:
            data_test.append(row)

    return True