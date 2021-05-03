# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Angeles Caballero Floro
"""

import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)


print('\n------------------------------------ EJERCICIO BONUS -------------------------------------\n')

#---------------------------------------------------------------------------------#
#-----------Funciones implementadas por los profesores de prácticas---------------#
#---------------------------------------------------------------------------------#


label4 = 1
label8 = -1

# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y


print('\n------------------------------------------------------------------------------------------')
print('Ejercicio 1:')
print('------------------------------------------------------------------------------------------')

#---------------------------------------------------------------------------------#
#-------------------------Funciones implementadas por mi--------------------------#
#---------------------------------------------------------------------------------#


# FUNCIONES PARA EL MODELO DE REGRESIÓN LINEAL  

# Funcion para calcular el error
def Err(x,y,w):
    y = y.reshape(-1,1)
    sumatoria = (x.dot(w)-y)**2 #numpy dot() -> Producto puntual de dos matrices.
    media = np.mean(sumatoria, axis = 0)  #numpy mean() -> Calcula la media aritmética
    media = media.reshape(-1,1)
    return media

#Derivada de la función error
def Edw(x,y,w):
    y = y.reshape(-1,1)
    sumatoria = (x.dot(w)-y)*x
    media = np.mean(sumatoria*2, axis = 0)
    media = media.reshape(-1,1)
    return media

# Gradiente Descendente Estocastico
def sgd( x, y, w, eta,tope, maxI ):
    iterations = 0
    while not Err(x, y, w) < tope and not iterations >= maxI: 
        w = w - eta*Edw(x,y,w)
        iterations += 1 
    return w

#Función que calcula una muess de tamaño M
def minibatch(x,y,M):
    index = np.random.choice(y.size, M, replace=False)
    x_minibatch = x[index,:]
    y_minibatch = y[index]
    
    return x_minibatch, y_minibatch

# FUNCIONES PARA EL AJUSTE CON PLA-POCKET 

def ajusta_PLA_para_pocket(datos, label, max_iter, vini):
    w = vini
    contador = 0
    j = 0
    while not j >= label.size and not contador >= 50:
        if (np.dot(np.transpose(w), datos[j]) * label[j])[0] <= 0:
            a = label[j]*datos[j]
            for i in range(0,2):
                w[i] = w[i]+a[i]
            contador = 0
        else:
            contador += 1
        j += 1
    i += 1
    #print("Para el valor inicial ",vini, " he necesitado ", i, " iteraciones.")
    return w


def PLA_pocket(datos, label, max_iter, w):
    w_mejor = ajusta_PLA_para_pocket(datos, label, max_iter, w)
    i=1
    for i in range(max_iter):
        w = ajusta_PLA_para_pocket(datos, label, max_iter, w)
        if(Err(datos, label, w) < Err(datos, label, w_mejor)):
            w_mejor=w
    
    return w_mejor


#---------------------------------------------------------------------------------#
#------------------------------Generación de datos--------------------------------#
#---------------------------------------------------------------------------------#

# Lectura de los datos de entrenamiento
x, y = readData('../../datos/X_train.npy', '../../datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('../../datos/x_test.npy', '../../datos/y_test.npy', [4,8], [-1,1])

# Variables necesarias para los calculos del SGD:
x_mini, y_mini = minibatch(x,y,128)
error2get = 1e-14
eta = 0.01  
initial_point = np.zeros((3,1))

print("Datos generados")

input("\n--- Pulsar tecla para continuar ---\n")

#---------------------------------------------------------------------------------#
#-----------------------------------GRAFICAS--------------------------------------#
#---------------------------------------------------------------------------------#

#-----------------------------------TRAINING--------------------------------------#

# Plot outputs sin separar
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

#----  APLICAMOS EL MODELO DE REGRESIÓN LINEAL ----
w_ini = sgd( x_mini, y_mini, initial_point, eta, error2get, 1)
print ('\nBondad del resultado previa a la mejora:')
print ("Ein: ", Err(x,y,w_ini))
print ("Eout: ", Err(x_test, y_test, w_ini))


input("\n--- Pulsar tecla para continuar ---\n")


# Plot outputs separación sin mejora (TRAINING)

#....Cargamos labelx y labely
sgdX = np.linspace(0, 1, y.size)
sgdY = (-w_ini[0] - w_ini[1]*sgdX) / w_ini[2]

# .... Mostramos la gráfica por pantalla
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Clasificación sin mejora (TRAINING)')
ax.set_xlim((0, 1))
plt.plot(sgdX, sgdY, 'k-', linewidth=2, label='SGD')
plt.legend()
plt.ylim(-7,1)
plt.show()


#---- APLICAMOS EL PLA-POCKET COMO MEJORA ----#
print("\nPuede tardar un poco...\n")

w = PLA_pocket(x, y, 1500, w_ini)

print ('\nBondad del resultado tras la mejora:')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))


input("\n--- Pulsar tecla para continuar ---\n")


# Plot outputs separación CON mejora (TRAINING)

#....Cargamos labelx y labely
plaX = np.linspace(0, 1, y.size)
plaY = (-w[0] - w[1]*sgdX) / w[2]

fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Clasificación con mejora(TRAINING)')
ax.set_xlim((0, 1))
plt.plot(plaX, plaY, 'k-', linewidth=2, label='POCKET')
plt.legend()
plt.ylim(-7,1)
plt.show()

# Plot outputs


#------------------------------------TEST-----------------------------------------#

# Plot outputs sin separar
fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

# Plot outputs separación CON mejora (TRAINING)

#....Cargamos labelx y labely
plaX = np.linspace(0, 1, y.size)
plaY = (-w[0] - w[1]*sgdX) / w[2]

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Clasificación (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.plot(plaX, plaY, 'k-', linewidth=2, label='POCKET')
plt.ylim(-7,1)
plt.show()
