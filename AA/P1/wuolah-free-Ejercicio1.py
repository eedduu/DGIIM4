# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:24:42 2019

@author: Angeles Caballero Floro

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print('\n------------------------------------------------------------------------------------------')
print(' \nEJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS: GRADIENTE DESCENDIENTE')
print('\n------------------------------------------------------------------------------------------')


# ---------------------------------------------------------------------------------
# APARTADO 1

def E(u, v):
    e = (np.power(u, 2) * (np.exp(v)) - (2 * (np.power(v, 2) * np.exp(-u)))) ** 2
    return e


# Derivada parcial de E con respecto a u:
def dEu(u, v):
    du = 2 * (np.power(u, 2) * (np.exp(v)) - (2 * (np.power(v, 2) * np.exp(-u)))) * (
            2 * u * (np.exp(v)) + 2 * (np.power(v, 2) * np.exp(-u)))
    return du


# Derivada parcial de E con respecto a v:
def dEv(u, v):
    dv = 2 * (np.power(u, 2) * (np.exp(v)) - (2 * (np.power(v, 2) * np.exp(-u)))) * (
            np.power(u, 2) * (np.exp(v)) - (4 * v * np.exp(-u)))
    return dv


# Gradiente de E:
def gradE(u, v):
    return np.array([dEu(u, v), dEv(u, v)])


# Cálculo del gradiente descendiente:
def gradient_descent_E(w, eta, tope, maxI):
    iterations = 0
    funcion_descenso = [] #Pequeña modificación con respecto al código original para almacenar los valores obtenidos.
    indices_descenso = []
    while not E(w[0], w[1]) < tope and not iterations >= maxI:
        funcion_descenso.append(E(w[0], w[1]))
        indices_descenso.append(iterations)
        w = w - eta * gradE(w[0], w[1])
        iterations += 1
    return w, iterations, funcion_descenso, indices_descenso


# ---------------------------------------------------------------------------------
# APARTADO 2

# Variables necesarias para los calculos del ejercicio 1:
eta = 0.01  # eta
maxIter = 10000  # Numero máximo de iteraciones para el gradiente descendiente.
error2get = 1e-14
initial_point = np.array([1.0, 1.0])
w, it, e_w, e_i= gradient_descent_E(initial_point, eta, error2get, maxIter)
print('\n------------------------------------------------------------------------------------------')
print('\nEjercicio 2:')
print('\n------------------------------------------------------------------------------------------')
print('\n *¿Cuántas iteraciones tarda el algoritmo en obtener por primera vez un valor de E(u; v) inferior a 10^-14? \n--> Solucion: ',
    it)
print('\n *¿En qué coordenadas (u, v) se alcanzó por primera vez un valor igual o menor a 10^-14? \n--> Solucion: (',
      w[0], ', ', w[1], ')')


# DISPLAY FIGURE

x = np.linspace(-30, 30, 50)    #Devuelve 50 muestras espaciadas uniformemente, calculadas en el intervalo [ -30, 30 ].
y = np.linspace(-30, 30, 50)    #Devuelve 50 muestras espaciadas uniformemente, calculadas en el intervalo [ -30, 30 ].
X, Y = np.meshgrid(x, y)        #Devuelve las matrices de coordenadas de los vectores de coordenadas.
Z = E(X, Y)  # E_w([X, Y])

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet')

min_point = np.array([w[0], w[1]])
min_point_ = min_point[:, np.newaxis]

plt.figure(1)
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='GD - Ejercicio 2. E(u, v)')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')

plt.show()

plt.figure(2)
plt.plot(e_i, e_w)
plt.title('Descenso de w en E(u, v) ')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
# ---------------------------------------------------------------------------------
# APARTADO 3 FUNCIONES

def F(x, y):
    return (x ** 2) + 2 * (y ** 2) + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


# Derivada parcial de F con respecto a x:
def dFx(x, y):
    return 2 * x + 2 * (2 * np.pi) * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)


# Derivada parcial de E con respecto a v:
def dFy(x, y):
    return 4 * y + 2 * np.sin(2 * np.pi * x) * (2 * np.pi) * np.cos(2 * np.pi * y)


# Gradiente de F:
def gradF(x, y):
    return np.array([dFx(x, y), dFy(x, y)])


# Cálculo del gradiente descendiente:
    
def gradient_descent_F(w, eta, maxI):
    iterations = 0
    funcion_descenso = [] #Pequeña modificación con respecto al código original para almacenar los valores obtenidos.
    indices_descenso = []
    while not iterations >= maxI:
        funcion_descenso.append(F(w[0], w[1]))
        indices_descenso.append(iterations)
        w = w - eta * gradF(w[0], w[1])
        iterations+=1
    return w, funcion_descenso, indices_descenso

#--------------------------------------------------------------------------------------------------------------

#APARTADO 3 A
print('\n\n------------------------------------------------------------------------------------------')
print('\nEjercicio 3: ')
print('\n------------------------------------------------------------------------------------------')
print('\nApartado A:\n ')

##### VARIBLES COMUNES PARA AMBOS CÁLCULOS
maxIter_F = 50  # Numero máximo de iteraciones para el gradiente descendiente.
initial_point_F = np.array([0.1, 0.1])


#---- 0.01 ----
eta = 0.01
w_1, f_w, i_w = gradient_descent_F(initial_point_F, eta, maxIter_F)

print('\n *Coordenadas (x, y) tras 50 iteraciones con una tasa de aprendizaje del 0.01 \n--> Solucion: (',
      w_1[0], ', ', w_1[1], ')')

# DISPLAY FIGURE 
plt.figure(3)
plt.plot(i_w, f_w)
plt.title('Descenso de w para una tasa de 0.01')
plt.show()

# DISPLAY FIGURE f(x,y)
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = F(X, Y)  # F_w_1([X, Y])

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet')

min_point = np.array([w_1[0], w_1[1]])
min_point_ = min_point[:, np.newaxis]

plt.figure(4)
ax.plot(min_point_[0], min_point_[1], F(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='GD - Ejercicio 3. F(x,y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('F(x,y)')

plt.show()


#---- 0.1 ----
initial_point_F = np.array([0.1, 0.1])
eta = 0.1
w_2, f_w, i_w = gradient_descent_F(initial_point_F, eta, maxIter_F)

print('\n *Coordenadas (x, y) tras 50 iteraciones con una tasa de aprendizaje del 0.1 \n--> Solucion: (',
      w_2[0], ', ', w_2[1], ')')

# DISPLAY FIGURE 
plt.figure(5)
plt.plot(i_w, f_w)
plt.title('Descenso de w para una tasa de 0.1 con 50 iteraciones')
plt.show()



# DISPLAY FIGURE EXAMPLE OF 100 ITERATIONS

#---- 0.01 ----
w_1, f_w, i_w = gradient_descent_F(initial_point_F, 0.01, 100)
print('\n *Coordenadas (x, y) tras 100 iteraciones con una tasa de aprendizaje del 0.01 \n--> Solucion: (',
      w_1[0], ', ', w_1[1], ')')

plt.figure(6)
plt.plot(i_w, f_w)
plt.title('Descenso de w para una tasa de 0.01')
plt.show()


#---- 0.1 ----
w_2, f_w, i_w = gradient_descent_F(initial_point_F, 0.1, 100)

print('\n *Coordenadas (x, y) tras 100 iteraciones con una tasa de aprendizaje del 0.1 \n--> Solucion: (',
      w_2[0], ', ', w_2[1], ')')

plt.figure(7)
plt.plot(i_w, f_w)
plt.title('Descenso de w para una tasa de 0.1 con 100 iteraciones')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
#--------------------------------------------------------------------------------------------------------------

#APARTADO 3 B
print(' \n------------------------------------------------------------------------------------------')

eta = 0.01

print('\nApartado B:\n ')
#---- (0.1, 0.1) ----
initial_point = np.array([0.1, 0.1])
w, f_w, i_w = gradient_descent_F(initial_point, 0.01, 50)
print('(0.1,0.1)\t', w[0], '\t', w[1], '\t', F(w[0],w[1]))

plt.figure(8)
plt.title('Comparación de las funciones')
plt.plot(i_w, f_w, 'r-', label= '(0.1, 0.1)')


#---- (1.0, 1.0) ----
initial_point = np.array([1.0, 1.0])
w, f_w, i_w = gradient_descent_F(initial_point, eta, 50)
print('(1.0,1.0)\t', w[0], '\t', w[1], '\t', F(w[0],w[1]))

plt.plot(i_w, f_w, 'b-', label= '(1.0, 1.0)')

#---- (-0.5, -0.5) ----

initial_point = np.array([-0.5, -0.5])
w, f_w, i_w = gradient_descent_F(initial_point, eta, 50)
print('(-0.5,-0.5)\t',  w[0], '\t', w[1], '\t', F(w[0],w[1]))

plt.plot(i_w, f_w, 'y-', label= '(-0.5, -0.5)')

#---- (-1.0, -1.0) ----
initial_point = np.array([-1.0, -1.0])
w, f_w, i_w = gradient_descent_F(initial_point, eta, 50)
print('(-1.0,-1.0)\t',  w[0], '\t', w[1], '\t', F(w[0],w[1]))

plt.plot(i_w, f_w, 'g-', label= '(-1.0, -1.0)')
plt.show()
