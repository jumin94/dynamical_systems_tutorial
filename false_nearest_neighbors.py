
import numpy as np 

def vecinos(y, dim):
    #y contiene los puntos del embedding [[P1], [P2], [P3],...,[PN]]
    #dim = la dimension del embedding
    #Definimos el numero de puntos
    N = len(y)
    #Transformamos el vector y en un array
    y = np.array(y)
    #Inicializamos la matriz de distancia cuadrática entre puntos i j
    Dij = []
    #Calculamos las distancias
    for k in range(len(y)-1):
        #Armamos una matriz de "N-k-1" filas y "dim" columnas en las que se
        #repite el punto Pk al que le vamos a calcular la distancia con el resto
        My = np.full((N-k-1, dim), y[k])
        if dim ==1:
            #Si la dimension es 1, la distancia es restar (Pk - Pj)^2
            Dktodos = np.power((My.transpose() - y[k+1:])[0], 2, dtype=float)
        else:
            #Si la dimension es >1, la distancia es ((xk-xj)^2+(yk-yj)^2+...)
            #por eso hacemos la resta de componentes, elevamos al cuadrado
            #y despues sumamos
            Dktodos = np.sum(np.power(My-y[k+1:],2, dtype=float), 1, dtype=float)
        #La fila de la matriz va a ser ceros hasta el lugar k y despues empiezan
        #las distancias que calculamos
        fila = list(np.concatenate((np.zeros(k+1), Dktodos)))
        Dij.append(fila)
        #print("Porcentaje :", 100*k/N)
    #Al ultimo punto no hace falta calcularle las distancias. Agregamos una fila
    #de ceros
    Dij.append(list(np.zeros(N)))
    Dij = np.array(Dij)
    #Hasta aca tenemos una matriz triangular superior, con ceros en la diagonal
    #La matriz debe ser simetrica. Y para que al buscar la minima distancia
    #no nos agarre el punto consigmo mismo; le sumamos algo grande a la diagonal    
    Dij = Dij + Dij.transpose() + np.eye(N) *100
    #Ya teniendo la matriz de distancias cuadraticas, buscamos los indices
    #en los que esta el minimo de cada fila
    indice_min = np.argmin(Dij, 1)
    #Teniendo los indices, guardamos la minima distancia
    dist_min = []
    for k in range(N):
        dist_min.append(np.sqrt(Dij[k][indice_min[k]]))
    #Por utlimo, devolvemos los indices de minimo y el valor de la distancia
    return indice_min, dist_min



def porcentaje_falsos_vecinos(x):
    porcentaje_falsos_vecinos = []
    dim = 1

    T = 50
    indice_min, dist_min = vecinos(x, dim)
    #Determino cuales son falsos vecinos para 1d
    R_crecimiento = []
    falsos_vecinos = 0
    puntos_noanalizados = 0
    for k in range(len(indice_min)-T):
        if indice_min[k]+dim*T < len(x):
            R_aux = np.abs( x[k+dim*T] - x[indice_min[k]+dim*T] ) / dist_min[k]
            R_crecimiento.append(R_aux)
            if R_aux >= 10:
                falsos_vecinos+=1
        else:
            puntos_noanalizados+=1

    porcentaje_falsos_vecinos.append(falsos_vecinos / (len(x)-T))
    
    y_emb_2 = []

    for k in range(len(x)-1*T):
        y_emb_2.append([x[k], x[k+T]])

    dim = 2
    indice_min, dist_min = vecinos(y_emb_2, dim)

    #Determino cuales son falsos vecinos para 2d
    R_crecimiento = []
    falsos_vecinos = 0
    puntos_noanalizados = 0
    for k in range(len(indice_min)-T):
        if indice_min[k]+dim*T < len(x):
            R_aux = np.abs( x[k+dim*T] - x[indice_min[k]+dim*T] ) / dist_min[k]
            R_crecimiento.append(R_aux)
            if R_aux >= 10:
                falsos_vecinos+=1
        else:
            puntos_noanalizados+=1
    print("El porcentaje de falsos vecinos es:", falsos_vecinos / (len(x)-T))
    print(puntos_noanalizados)
    porcentaje_falsos_vecinos.append(falsos_vecinos / (len(x)-T))

    y_emb_3 = []

    for k in range(len(x)-2*T):
        y_emb_3.append([x[k], x[k+T], x[k+2*T]])

    dim = 3
    indice_min, dist_min = vecinos(y_emb_3, dim)
    #Determino cuales son falsos vecinos para 3d
    R_crecimiento = []
    falsos_vecinos = 0
    puntos_noanalizados = 0
    for k in range(len(indice_min)-T):
        if indice_min[k]+dim*T < len(x):
            R_aux = np.abs( x[k+dim*T] - x[indice_min[k]+dim*T] ) / dist_min[k]
            R_crecimiento.append(R_aux)
            if R_aux >= 10:
                falsos_vecinos+=1
        else:
            puntos_noanalizados+=1
    print("El porcentaje de falsos vecinos es:", falsos_vecinos / (len(x)-T))
    print(puntos_noanalizados)
    porcentaje_falsos_vecinos.append(falsos_vecinos / (len(x)-T))


    y_emb_4 = []

    for k in range(len(x)-3*T):
        y_emb_4.append([x[k], x[k+T], x[k+2*T], x[k+3*T]])

    dim = 4
    indice_min, dist_min = vecinos(y_emb_4, dim)

    #Determino cuales son falsos vecinos para 4d
    R_crecimiento = []
    falsos_vecinos = 0
    puntos_noanalizados = 0
    for k in range(len(indice_min)-T):
        if indice_min[k]+dim*T < len(x):
            R_aux = np.abs( x[k+dim*T] - x[indice_min[k]+dim*T] ) / dist_min[k]
            R_crecimiento.append(R_aux)
            if R_aux >= 10:
                falsos_vecinos+=1
        else:
            puntos_noanalizados+=1
            
    porcentaje_falsos_vecinos.append(falsos_vecinos / (len(x)-T))

    y_emb_5 = []

    for k in range(len(x)-4*T):
        y_emb_5.append([x[k], x[k+T], x[k+2*T], x[k+3*T], x[k+4*T]])

    dim = 5
    indice_min, dist_min = vecinos(y_emb_5, dim)
    #Determino cuales son falsos vecinos para 4d
    R_crecimiento = []
    falsos_vecinos = 0
    puntos_noanalizados = 0
    for k in range(len(indice_min)-T):
        if indice_min[k]+dim*T < len(x):
            R_aux = np.abs( x[k+dim*T] - x[indice_min[k]+dim*T] ) / dist_min[k]
            R_crecimiento.append(R_aux)
            if R_aux >= 10:
                falsos_vecinos+=1
        else:
            puntos_noanalizados+=1

    porcentaje_falsos_vecinos.append(falsos_vecinos / (len(x)-T))
    return porcentaje_falsos_vecinos


def embedding(tau, dato, w):
    """
    Generalized embedding function.

    Parameters:
    - tau: time delay
    - dato: 1D numpy array (signal)
    - w: embedding dimension

    Returns:
    - embedding: 2D numpy array of shape (N, w)
    """
    if len(dato.shape) != 1:
        raise ValueError("Input 'dato' must be a 1D array.")

    max_index = dato.shape[0] - (w - 1) * (tau + 1)
    if max_index <= 0:
        raise ValueError("Embedding dimension and tau too large for data length.")

    indices = (np.arange(w) * (tau + 1)) + np.arange(max_index).reshape(-1, 1)
    embedding = dato[indices]

    return embedding  # shape (N, w)

    