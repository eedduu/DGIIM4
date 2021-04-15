# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:17:26 2019

@author: Angeles Caballero Floro
"""
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(1)

print('\n\n------------------------------------------------------------------------------------------')
print('\nEJERCICIO SOBRE REGRESION LINEAL\n')
print('\n------------------------------------------------------------------------------------------')


# ---------------------------------------------------------------------------------
# EJERCICIO 1: Implementación de las funciones

etiqueta_5 =  1
etiqueta_1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
    # Leemos los ficheros    
    datax = np.load(file_x)
    datay = np.load(file_y)
    y = []
    x = []    
    # Solo guardamos los datos cuya clase sea la 1 o la 5
    for i in range(0,datay.size):
        if datay[i] == 5 or datay[i] == 1:
            if datay[i] == 5:
                y.append(etiqueta_5)
            else:
                y.append(etiqueta_1)
            x.append(np.array([1, datax[i][0], datax[i][1]]))
            
    x = np.array(x, np.float64)
    y = np.array(y, np.float64)
    
    return x, y

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

#Función que calcula una muestra de tamaño M
def minibatch(x,y,M):
    index = np.random.choice(y.size, M, replace=False)
    x_minibatch = x[index,:]
    y_minibatch = y[index]
    
    return x_minibatch, y_minibatch

def pseudoinverse(x,y):
    return (np.linalg.pinv(x.T.dot(x)).dot(x.T)).dot(y)

# ---------------------------------------------------------------------------------
# APARTADO 1: Operaciones

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

# Variables necesarias para los calculos del SGD:
x_mini, y_mini = minibatch(x,y,128)
error2get = 1e-14
eta = 0.01  
initial_point = np.zeros((3,1))

print('\n\n------------------------------------------------------------------------------------------')
print('\nEJERCICIO 1:\n')
print('\n------------------------------------------------------------------------------------------')

#---- GRADIIENTE DESCENDENTE ESTOCÁSTICO ----
w = sgd( x_mini, y_mini, initial_point, eta, error2get, 2000)

print ('\nBondad del resultado para grad. descendente estocastico:')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

# DISPLAY FIGURE
sgdX = np.linspace(0, 1, y.size)
sgdY = (-w[0] - w[1]*sgdX) / w[2]

plt.figure(1)
plt.scatter(x[:,1], x[:,2], c=y)
plt.plot(sgdX, sgdY, 'r-', linewidth=2, label='SGD')

plt.title('\nREGRESION - Ejercicio 1. Modelo de regresión lineal con el SGD')
plt.ylabel('Intensidad promedio')
plt.xlabel('Simetria')

#---- PSEUDOINVERSA ----
wpinv = pseudoinverse( x, y)
wpinv = wpinv.reshape((3,1))

print ('\nBondad del resultado para pseudo-inversa:')
print ("Ein: ", Err(x,y,wpinv))
print ("Eout: ", Err(x_test, y_test, wpinv))

# DISPLAY FIGURE
pinvX = np.linspace(0, 1, y.size)
pinvY = (-wpinv[0] - wpinv[1]*pinvX) / wpinv[2]

plt.figure(2)
plt.scatter(x[:,1], x[:,2], c=y)
plt.plot(pinvX, pinvY, 'b-', linewidth=2, label='Pseudoinversa')

plt.title('\nREGRESION - Ejercicio 1. Modelo de regresión lineal con Pseudo-inversa')
plt.ylabel('Intensidad promedio')
plt.xlabel('Simetria')

#---- COMPARACIÓN DE AMBAS ----

plt.figure(3)
plt.scatter(x[:,1], x[:,2], c=y)
plt.plot(sgdX, sgdY, 'r-', linewidth=2, label='SGD')
plt.plot(pinvX, pinvY, 'b-', linewidth=2, label='Pseudoinversa')

plt.title('\nREGRESION - Ejercicio 1. Modelo de regresión lineal comparacion de SGD con Pseudo-inversa')
plt.ylabel('Intensidad promedio')
plt.xlabel('Simetria')
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")
# ---------------------------------------------------------------------------------
# EJERCICIO 2
# ---------------------------------------------------------------------------------
# APARTADO A: Implementación de las funciones

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
    return np.random.uniform(-size,size,(N,d))

# ---------------------------------------------------------------------------------
# APARTADO A: Cálculos
    
print('\n\n------------------------------------------------------------------------------------------')
print('\nEjercicio 2: ')
print('\n------------------------------------------------------------------------------------------')

A = simula_unif(1000, 2, 1)

# DISPLAY FIGURE
plt.figure(4)
plt.scatter(A[:,0], A[:,1]) #Para puntos
plt.title('REGRESION - Ejercicio 2. EXPERIMENTO APARTADO A')
plt.show()  #Muestra una grafica de puntos aleatorios

input("\n--- Pulsar tecla para continuar ---\n")
# ---------------------------------------------------------------------------------
# APARTADO B: Implementación de las funciones
def f(x, y):
    f = []
    for i in range(x.size):
        if  ((x[i]-0.2)**2 + y[i]**2 - 0.6) >= 0:
            f.append(1)
        else:
            f.append(-1)
    f = np.array(f)
    f = f.reshape(-1,1)
    return f

def f_Ruido(f, porcentaje):
    index = np.random.choice(f.size,  size = int(porcentaje*f.size), replace=False)
    for i in range(f.size):
        if np.isin(i,index):
            f[i] = -f[i]
    f = np.array(f)
    f = f.reshape(-1,1)
    return f

# ---------------------------------------------------------------------------------
# APARTADO B: Calculos y Gráficas

# Plot outputs sin ruido
B = f(A[:,0],A[:,1])
plt.figure(5)
plt.scatter(A[:,0],A[:,1],c = B[:,0]) #Para puntos
plt.title('REGRESION - Ejercicio 2. EXPERIMENTO APARTADO B- SIN RUIDO')
plt.show()  #Muestra una grafica de puntos aleatorios

# Plot outputs con ruido
B_ruido = f_Ruido(B,0.1)
plt.figure(6)
plt.scatter(A[:,0],A[:,1],c = B_ruido[:,0]) #Para puntos
plt.title('REGRESION - Ejercicio 2. EXPERIMENTO APARTADO B- CON RUIDO')
plt.show()  #Muestra una grafica de puntos aleatorios
input("\n--- Pulsar tecla para continuar ---\n")

# ---------------------------------------------------------------------------------
# APARTADO C: Calculos y Gráficas
C = np.c_[np.ones((1000, 1), np.float64), A]

x = C
y = B_ruido
w = sgd(x, y, initial_point, eta, error2get, 200 )
print('\nApartado C:\n ')
print ("Ein: ", Err(x,y,w))

# Plot outputs
lineX = np.linspace(-1, 1, y.size)
lineY = (-w[0] - w[1]*lineX) / w[2]

plt.figure(7)

plt.scatter(x[:,1], x[:,2], c=y[:,0])
plt.plot(lineX, lineY, 'r-', linewidth=2)

plt.title('REGRESION - Ejercicio 2. EXPERIMENTO APARTADO C')
plt.ylabel('Intensidad promedio')
plt.xlabel('Simetria')
plt.ylim(-1.0,1.0)
input("\n--- Pulsar tecla para continuar ---\n")

# ---------------------------------------------------------------------------------
# APARTADO D: Implementación de las funciones
def repABC(N):
    EinM = 0;
    EoutM = 0;
    
    for i in range (N):
        A = simula_unif(1000, 2, 1)
        B = f(A[:,0],A[:,1]) 
        C = np.c_[np.ones((1000, 1), np.float64), A]
    
        A_test = simula_unif(1000, 2, 1)
        B_test = f(A_test[:, 0], A_test[:, 1])
        B_test = f_Ruido(B_test,0.1)
        C_test = np.c_[np.ones((1000, 1), np.float64), A_test]
    
        w = sgd(C, B, np.zeros((3,1)), 0.01,error2get, 200)
        EinM = EinM + Err(C, B, w)                  # Vamos acumulando los valores
        EoutM = EoutM + Err(C_test, B_test, w)      # en estas vaiables
        
        #Descomentar para ver el calculo de la media paso a paso.
        #print ("\nIteracion", i)  
        #print ("Ein: ", Err(C,B,w))
        #print ("Eout: ", Err(C_test, B_test, w))
        #print ("Media Ein: ", EinM/i)
        #print ("Media Eout: ", EoutM/i)
        
    print ("Media Ein: ", EinM/i)
    print ("Media Eout: ", EoutM/i)

# ---------------------------------------------------------------------------------
# APARTADO D: Calculos 
print('\nApartado D:\n ')
repABC(1000)
