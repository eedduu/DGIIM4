# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Angeles Caballero Floro
"""

import numpy as np
import matplotlib.pyplot as plt



# Fijamos la semilla
np.random.seed(1)

print('\n------------------------------------ MODELOS LINEALES ------------------------------------\n')


#---------------------------------------------------------------------------------#
#-----------Funciones implementadas por los profesores de prácticas---------------#
#---------------------------------------------------------------------------------#


# Función que nos permite simular la pendiente y la recta de una recta

def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b


# Función que nos permite simular los elementos de un conjunto de datos con 
# distribución uniforme
    
def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))


# Función que devuelve el signo (+ o -) de un dato 

def signo(x):
	if x >= 0:
		return 1
	return -1


# Función que facilita la clasificación (etiquetas) de los elementos de un conjunto
# de datos

def f(x, y, a, b):
	return signo(y - a*x - b)


###################################################################################

# EJERCICIO 2.1 - A: ALGORITMO PERCEPTRON

print('\n------------------------------------------------------------------------------------------')
print('Ejercicio 1:')
print('------------------------------------------------------------------------------------------')


#---------------------------------------------------------------------------------#
#-------------------------Funciones implementadas por mi--------------------------#
#---------------------------------------------------------------------------------#
    

def ajusta_PLA(datos, label, max_iter, vini):
    w = vini
    contador = 0
    i = 0
    while not i >= max_iter and not contador >= 50:
        j = 0
        while not j >= len(datos[:,0]) and not contador >= 50:
            if(np.dot(np.transpose(w), datos[j]) * label[j] <= 0):
                w = w + label[j]*datos[j]
                contador = 0
            else:
                contador += 1
            j += 1
        i += 1
    #print("Para el valor inicial ",vini, " he necesitado ", i, " iteraciones.")
    return w, i


#---------------------------------------------------------------------------------#
#------------------------------Generación de datos--------------------------------#
#---------------------------------------------------------------------------------#

# Calculamos X
X = simula_unif(50, 2, [-50,50])

print("Conjunto X de 50 elementos, generado satisfactoriamente...")


# Calculamos Y

#....Generamos una recta caracterizada por los valores a (pendiente) y b (término
#....independiente)
a, b = simula_recta([-50,50])

#....Creamos un array vacío
Y = []

for coordenadas in X:
    #....Generamos la etiqueta de cada uno de los valores de X, con la función 
    #....f(x, y, a, b) 
    etiqueta = f(coordenadas[0],coordenadas[1],a,b)
    #....y la añadimos al vector de etiquetas
    Y.append(etiqueta)    

Y = np.array(Y, np.float64)

print("Etiquetado de los elementos completado correctamente...")

# Añadimos una columna de unos a nuestra matriz X, para poder multiplicar por 
# el valor traspuesro de X en la función "ajusta_PLA"
X = np.c_[np.ones((50, 1), np.float64), X]


#---------------------------------------------------------------------------------#
#-------------------------Empleo de los datos generados---------------------------#
#---------------------------------------------------------------------------------#


#Ejecutar el algoritmo PLA con los datos simulados con el vector cero

#.... Inicializamos el punto de partida a (0,0,0)
initial_point= np.zeros(3)
w, contador = ajusta_PLA(X,Y,500,initial_point)


#Ejecutar el algoritmo PLA con los datos simulados con el vector cero

#.... Vector en el que almacenaremos las iteraciones necesarias para converger 
#.... para cada uno de los datos de partida generados aleatoriamente
iterations = []

#.... Vector en el que almacenaremos los datos de partida generados aleatoriamente
#.... para el siguiente apartado
randoms = []

for i in range(0,10):
    #.... Inicializamos el punto de partida a valores aleatorios entre 0 y 1
    initial_point = np.random.rand(3)
    #.... lo guardamos para el apartado 2b
    randoms.append(initial_point)
    w, contador = ajusta_PLA(X,Y,500,initial_point)
    #.... Almacenamos el numero de iteraciones necesarias para converger, para 
    #.... calcular la media de los 10 puntos.
    iterations.append(contador)

    
print('\n- A - Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))


input("--- Pulsar tecla para continuar ---\n")


# EJERCICIO 2.1 - B: ALGORITMO PERCEPTRON

#---------------------------------------------------------------------------------#
#-------------------------Funciones implementadas por mi--------------------------#
#---------------------------------------------------------------------------------#


# Con esta función, obtenemos los indices de los elementos con los que le meteremos 
# cierto porcentaje de ruido a una muestra

def ruido (Y,porcentaje):
    # Separamos los elementos que son positivos de los negativos y guardamos sus
    # indices en vectores separados
    Y_pos = np.where(Y > 0)
    Y_neg = np.where(Y < 0)
    
    Y_pos = np.array(Y_pos)
    Y_neg = np.array(Y_neg)
    
    # Seleccionamos una muestra del tamaño especificado
    index_pos = np.random.choice(Y_pos.size,  size = int(porcentaje*Y_pos.size), replace=False)
    index_neg = np.random.choice(Y_neg.size,  size = int(porcentaje*Y_neg.size), replace=False)
    
    index_pos = np.array(index_pos)
    index_neg = np.array(index_neg)

    return Y_pos, Y_neg, index_pos, index_neg


#---------------------------------------------------------------------------------#
#------------------------------Generación de datos--------------------------------#
#---------------------------------------------------------------------------------#

#Obtenemos los vectores necesarios para introducir el ruido en la muestra    
Y_pos, Y_neg, index_pos, index_neg = ruido (Y, 0.1)

# Invertimos las etiquetas de los indices que acabamos de seleccionar
for i in range(Y.size):
    if np.isin(i,Y_pos[0][index_pos]):
        Y[i] = -Y[i]
    
for i in range(Y.size):
    if np.isin(i,Y_neg[0][index_neg]):
        Y[i] = -Y[i]

# Con esto obtenemos una nueva variable Y en la que hemos simulado un 10% de ruido

print("Ruido introducido en la muestra de forma adecuada...")

#---------------------------------------------------------------------------------#
#-------------------------Empleo de los datos generados---------------------------#
#---------------------------------------------------------------------------------#

#Ejecutar el algoritmo PLA con los datos simulados con el vector cero

#.... Inicializamos el punto de partida a (0,0,0)
initial_point= np.zeros(3)
w, contador = ajusta_PLA(X,Y,500,initial_point)


#Ejecutar el algoritmo PLA con los datos simulados con el vector cero

#.... Vector en el que almacenaremos las iteraciones necesarias para converger 
#.... para cada uno de los datos de partida generados aleatoriamente
iterations2 = []

for i in range(0,10):
    #.... Seleccionamos el valor aleatorio i de los generados en el apartado enterior
    initial_point = randoms[i]
    w, contador = ajusta_PLA(X,Y,500,initial_point)
    #.... Almacenamos el numero de iteraciones necesarias para converger, para 
    #.... calcular la media de los 10 puntos.
    iterations2.append(contador)
    

print('\n- B - Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations2))))

input("\n--- Pulsar tecla para continuar ---\n")

##################################################################################
##################################################################################
##################################################################################

# EJERCICIO 2.2: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

print('\n------------------------------------------------------------------------------------------')
print('Ejercicio 2:')
print('------------------------------------------------------------------------------------------')

#---------------------------------------------------------------------------------#
#-------------------------Funciones implementadas por mi--------------------------#
#---------------------------------------------------------------------------------#


def gradient_sigmoid(x, y, w):
    return -(y * x)/(1 + np.exp(y * w.dot(x.reshape(-1,))))


# Función para ajustar un clasificador basado en regresión lineal mediante
# el algoritmo SGD

def sgdRL(x_data, y_data, w, dif_min=0.01, tasa=0.01):
    
    # Obtenemos número de elementos
    N = x_data.shape[0]
    
    # Establecemos una diferencia inicial entre w(t-1) y w(t)
    delta = np.inf
    
    # Mientras la diferencia sea superior al umbral...
    while delta > dif_min:
        # Creaos una nueva permutaciónde los datos
        indexes = np.random.permutation(N)
        x_data = x_data[indexes, :]
        y_data = y_data[indexes]
        
        # Guardamos w(t-1)
        prev_w = np.copy(w)
        
        # Actualizaos w
        for x, y in zip(x_data, y_data):
            w = w - tasa * gradient_sigmoid(x, y, w)
        
        # Actualizamos delta
        delta = np.linalg.norm(prev_w - w)
    
    return w.reshape(-1,)


# Funcion para calcular el error

def Err(x,y,w):
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)

    z = (x.dot(w))
    Err = np.log(1 + np.exp(-(y*z)))
    return np.mean(Err, axis=0)


#Función que calcula una muess de tamaño M
    
def muestras(x,y,M):
    x_test = []
    y_test = []
    
    #.... En este caso divido nuestro conjunto en 3 partes, 1 para el entrenamiento
    #.... y las restantes para el test
    for i in range(3):
        #.... Seleccionamos de forma aleatoria un tercio de los datos sin repetir
        index = np.random.choice(y.size, int(M*y.size), replace=False)
        #.... Rellenamos nuestros conjutos
        if i == 0:
            x_train = x[index,:]
            y_train = y[index]
        else:
            for ind in index:
                x_test.append(x[ind])
                y_test.append(y[ind])
                
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_test, y_test, x_train, y_train


#---------------------------------------------------------------------------------#
#------------------------------Generación de datos--------------------------------#
#---------------------------------------------------------------------------------#

# Añadimos una columna de unos a nuestra matriz X, para poder multiplicar por 
# el valor traspuesro de X en la función "ajusta_PLA"
X = np.c_[np.ones((50, 1), np.float64), X]

# Calculamos X
X = simula_unif(1500, 2, [0,2])

print("Conjunto X de 1500 elementos, generado satisfactoriamente...")

#....Generamos una recta caracterizada por los valores a (pendiente) y b (término
#....independiente)
a, b = simula_recta([0,2])


# Calculamos Y
#....Creamos un array vacío
Y = []

for coordenadas in X:
    #....Generamos la etiqueta de cada uno de los valores de X segun una nueva f
    f = coordenadas[1] - a*coordenadas[0] - b
    if(1/(1 + np.exp(f))) >= 0.5:
        etiqueta = 1
    else:
        etiqueta = -1
    #....y la añadimos al vector de etiquetas
    Y.append(etiqueta)    

Y = np.array(Y, np.float64)


print("Etiquetado de los elementos completado correctamente...")

x_test, y_test, x_train, y_train = muestras(X,Y,0.2)

print("Conjuntos de entrenamiento y examen separados...")


#---------------------------------------------------------------------------------#
#-------------------------Empleo de los datos generados---------------------------#
#---------------------------------------------------------------------------------#

#-----------------------------------TRAINING--------------------------------------#

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


print("\nPuede tardar un poco...\n")


# Añadimos una columna de unos a nuestra matriz X, para poder multiplicar 
x_train = np.c_[np.ones((y_train.size, 1), np.float64), x_train]


#.... Inicializamos el punto de partida a (0,0,0)
initial_point= np.zeros(3)
w = sgdRL(x_train, y_train, initial_point, dif_min=0.01, tasa=0.01)

print ("- Ein: ", Err(x_train, y_train, w))


# Plot outputs

#....Cargamos labelx y labely
sgdLR_x = np.linspace(0, 2, y_train.size)
sgdLR_y = (-w[0] - w[1]*sgdLR_x) / w[2]

# .... Mostramos la gráfica por pantalla
plt.figure(1)
plt.title('EJERCICIO 2.2 - CONJUNTO DE ENTRENAMIENTO')
plt.scatter(x_train[:,1], x_train[:,2], c=y_train)
plt.plot(sgdLR_x, sgdLR_y, 'r-', linewidth=2)
plt.xlim(0,2)
plt.ylim(0,2)
plt.show()

#------------------------------------TEST-----------------------------------------#

# Añadimos una columna de unos a nuestra matriz X, para poder multiplicar
x_test = np.c_[np.ones((y_test.size, 1), np.float64), x_test]

print ("- Eout: ", Err(x_test, y_test, w))


# Plot outputs

# .... Mostramos la gráfica por pantalla
plt.figure(2)
plt.title('EJERCICIO 2.2 - CONJUNTO DE TEST')
plt.scatter(x_test[:,1], x_test[:,2], c=y_test)
plt.plot(sgdLR_x, sgdLR_y, 'r-', linewidth=2)
plt.xlim(0,2)
plt.ylim(0,2)
plt.show()