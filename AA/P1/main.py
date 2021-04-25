# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Eduardo Morales Muñoz
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')

#Ejercicio 1

#Implementación del algoritmo de gradiente descendente
def gradient_descent(pto, lr, maxit, function, grad, error):
    iteraciones = 0
    while iteraciones<maxit and function(pto[0], pto[1])>error:
        iteraciones+=1
        pto = pto-lr*grad(pto[0], pto[1])     
    
    return pto, iteraciones

print('Ejercicio 2\n')

#Función
def E(u,v):
    return (u**3 * np.exp(v-2) - 2*v**2*np.exp(-u))**2  

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2*(u**3*np.exp(v-2) - 2*v**2*np.exp(-u)) * (3*u**2*np.exp(v-2)+2*np.exp(-u)*v**2)

    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*(u**3 * np.exp(v-2) - 4 * np.exp(-u)*v) * (u**3 * np.exp(v-2) - 2*np.exp(-u) * v**2)


#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])


#Parámetros para el gradiente descendente
eta = 0.1
maxIter = 10000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])


w, it = gradient_descent(initial_point, eta, maxIter, E, gradE, error2get)


print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')


# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#--------------------------------------------

print('Ejercicio 3\n')

#Función
def f(x,y):
    return (x+2)**2 + 2*(y-2)**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

#Derivada parcial respecto de x
def fx(x,y):
    return 2*(2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y) + x +2)
    
#Derivada parcial respecto de y
def fy(x,y):
    return 4*(np.pi*np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + y -2)

#Gradiente de la función
def gradf(x,y):
    return np.array([fx(x,y), fy(x,y)])

#Modificación del algoritmo de gradiente descendente para que guarde los valores de w
def gradient_descent_2(pto, lr, maxit, function, grad, error):
    iteraciones = 0
    w = [function(pto[0], pto[1])]
    while iteraciones<maxit and function(pto[0], pto[1])>error:
        iteraciones+=1
        pto = pto-lr*grad(pto[0], pto[1])  
        w.append(function(pto[0], pto[1]))
    
    return pto, iteraciones, w

#Pongo los parámetros y hago los cálculos
pto_ini = np.array([-1.0, 1.0])
maxIt = 50

print('Con learning rate = 0.1\n')
lr = 0.1
w2, it2, arr= gradient_descent_2(pto_ini, lr, maxIt, f, gradf, -10000)

print ('Coordenadas obtenidas: (', w2[0], ', ', w2[1],')\n')

#Pinto la gráfica
plt.plot(np.arange(51), arr)
plt.title('w para un learning rate de 0.1')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


#Cambio el learning rate y cálculos
print('Con learning rate = 0.01\n')
lr = 0.01
w2, it2, arr = gradient_descent_2(pto_ini, lr, maxIt, f, gradf, -10000)

#Pinto la gráfica
plt.plot(np.arange(51), arr)
plt.title('w para un learning rate de 0.01')
plt.show()

print ('Coordenadas obtenidas: (', w2[0], ', ', w2[1],')\n')


input("\n--- Pulsar tecla para continuar ---\n")

#Para los diferentes puntos iniciales calculo las coordenadas finales
puntos = np.array([[-0.5, -0.5], [1.0,1.0], [2.1,-2.1], [-3,3], [-2,2]])

print('Para un learning rate de 0.01:\n')
for pto in puntos:
    fin, ite, valores = gradient_descent_2(pto, lr, maxIt, f, gradf, -10000)
    print('Para la coordenada inicial (',pto[0], ', ', pto[1], ') el valor mínimo es ', f(fin[0], fin[1]) , ' y las coordenadas finales son ( ', fin[0], ' , ', fin[1], ' ) \n'  )


# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w2[0],w2[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], f(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.3. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
 	# Leemos los ficheros	
 	datax = np.load(file_x)
 	datay = np.load(file_y)
 	y = []
 	x = []	
 	# Solo guardamos los datos cuya clase sea la 1 o la 5
 	for i in range(0,datay.size):
         if datay[i]==5 or datay[i]==1:
             if datay[i] ==5:
                 y.append(label5)
             else:
                 y.append(label1)
             x.append(np.array([1, datax[i][0], datax[i][1]]))
 			
 	x = np.array(x, np.float64)
 	y = np.array(y, np.float64)
 	
 	return x, y

# Funcion para calcular el error
def Err(x,y,w):
    y = y.reshape(-1,1)
    suma = (x.dot(w)-y)**2
    media = np.mean(suma, axis = 0)  
    media.reshape(-1,1)
    return media

# Derivada de la función de error
def ErrDeriv(x, y, w):
    y = y.reshape(-1, 1)
    suma = (x*(x.dot(w)-y))
    media = np.mean(2*(suma), axis=0)    
    media = media.reshape(-1,1)
    return media

# Gradiente Descendente Estocastico
def sgd(x, y, w, lr, maxIt, error, M):
    iteraciones = 0
    x_b, y_b = x, y
    while iteraciones < maxIt and Err(x_b, y_b, w) > error:
        iteraciones+=1
        x_b, y_b = batch(x, y, M) 
        w = w - lr*ErrDeriv(x_b, y_b, w)
        
    return w

def batch(x,y,M):
    ind = np.random.choice(y.size, M, replace=False)
    x_b = x[ind, :]
    y_b = y[ind]
    
    return x_b, y_b

# Pseudoinversa	
def pseudoinverse(x, y):
    return (np.linalg.pinv(x.T.dot(x)).dot(x.T)).dot(y)


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


#Parámetros para sgd
pto_ini = np.zeros((3,1))
lr = 0.01
maxIt=200
tam_batch = int(x.size/maxIt)
error2get = 1e-14
x_b, y_b = batch(x, y, tam_batch)


w = sgd(x, y, pto_ini, lr, maxIt, error2get, tam_batch)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))


sgdX = np.linspace(0, 1, y.size)
sgdY = (-w[0] - w[1]*sgdX) / w[2]


plt.scatter(x[:,1], x[:,2], c=y)
plt.plot(sgdX, sgdY, 'r-', label='SGD')

plt.title('Descenso de Gradiente Estocástico')
plt.ylabel('Intensidad promedio')
plt.xlabel('Simetria')
plt.show()

######PSEUDOINVERSA


wp = pseudoinverse(x, y)
wp = wp.reshape(3,1)  # Usando wp.reshape no me hacía el cambio

print ('\nBondad del resultado para pseudoinversa:\n')
print ("Ein: ", Err(x,y,wp))
print ("Eout: ", Err(x_test, y_test, wp))

pinvX = np.linspace(0, 1, y.size)
pinvY = (-wp[0] - wp[1]*pinvX) / wp[2]

plt.scatter(x[:,1], x[:,2], c=y)
plt.plot(pinvX, pinvY, 'b-', label='Pseudoinversa')

plt.title('PseudoInversa')
plt.ylabel('Intensidad promedio')
plt.xlabel('Simetria')
plt.show()

plt.scatter(x[:,1], x[:,2], c=y)
plt.plot(pinvX, pinvY, 'b-', label='Pseudoinversa')
plt.plot(sgdX, sgdY, 'r-', label='SGD')
plt.title('Comparativa')
plt.ylabel('Intensidad promedio')
plt.xlabel('Simetria')
plt.legend()
plt.show()


input("\n--- Pulsar intro para continuar ---\n")



print('Ejercicio 2\n')


# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
 	return np.random.uniform(-size,size,(N,d))

def sign(x):
 	if x >= 0:
         return 1
 	return -1

def f(x1, x2):
    a = (x1-0.2)**2
    b = x2**2
    return sign(a+b-0.6)


print('Apartado A.\n')
tam = 1000
muestra = simula_unif(tam, 2, 1)

plt.scatter(muestra[:,0], muestra[:,1])
plt.show()

input("\n--- Pulsar intro para continuar ---\n")


print('Apartado B\n')
#Creo un array con las imagenes de la muestra
f_muestra = []
for i in range(tam):
    f_muestra.append(f(muestra[:,0][i], muestra[:,1][i]))
f_muestra = np.array(f_muestra)
f_muestra = f_muestra.reshape(-1,1)

#Dibujo las imagenes de la muestra (sin ruido)
plt.scatter(muestra[:,0], muestra[:,1], c=f_muestra[:,0])
plt.title('Sin ruido')
plt.show()

#Añado ruido, seleccionando 10% de elementos al azar y cambiando el signo
f_ruido = f_muestra.copy()
porcentaje = 0.1
indices_r = np.random.choice(f_ruido.size, int(porcentaje*f_ruido.size), replace=False)
for i in range(f_ruido.size):
    if np.isin(i, indices_r):
        f_ruido[i]=-f_ruido[i]
        
#Muestro la imagen de la muestra (con ruido)
plt.scatter(muestra[:,0], muestra[:,1], c=f_ruido[:,0])
plt.title('Con ruido')
plt.show()

input("\n--- Pulsar intro para continuar ---\n")
print('Apartado c\n')
#Cálculos
x = np.c_[np.ones((1000,1)), muestra]
y = f_ruido
w = sgd(x,y, pto_ini, lr, maxIt, error2get, tam_batch)


#Dibujo la gráfica
X = np.linspace(-1, 1, y.size)
Y = (-w[0] - w[1]*X) / w[2]


plt.scatter(x[:,1], x[:,2], c=y[:,0])
plt.plot(X, Y, 'r-')

plt.title('Ajuste con un modelo lineal')
plt.ylabel('Intensidad promedio')
plt.xlabel('Simetria')
plt.ylim(-1.0,1.0)
plt.show()

print ("Ein: ", Err(x,y,w))

##----------

input("\n--- Pulsar intro para continuar ---\n")
print('Apartado d\n')
#Parámetros
Ein = 0
Eout = 0
N = 1000
tam = 1000
for i in range(N):
    #Hago la muestra
    np.random.seed(i)
    muestra = simula_unif(tam, 2, 1)
    f_muestra = []
    for i in range(tam):
        f_muestra.append(f(muestra[:,0][i], muestra[:,1][i]))
    f_muestra = np.array(f_muestra)
    f_muestra = f_muestra.reshape(-1,1)
    
    #Añado ruido a la muestra
    f_ruido = f_muestra.copy()
    porcentaje = 0.1
    indices_r = np.random.choice(f_ruido.size, int(porcentaje*f_ruido.size), replace=False)
    for i in range(f_ruido.size):
        if np.isin(i, indices_r):
            f_ruido[i]=-f_ruido[i]
    
    #Sumo a Ein
    x = np.c_[np.ones((1000,1)), muestra]
    y = f_ruido
    w = sgd(x, y, pto_ini, lr, maxIt, error2get, tam_batch)   
    Ein += Err(x,y, w)
    
    test = simula_unif(tam, 2, 1)
    x_t = np.c_[np.ones((1000,1)), test]
    y_t = []
    for i in range(tam):
        y_t.append(f(x_t[:,0][i], x_t[:,1][i]))
    y_t = np.array(y_t)
    y_t = y_t.reshape(-1,1)
    indices_r = np.random.choice(y_t.size, int(porcentaje*y_t.size), replace=False)
    for i in range(y_t.size):
        if np.isin(i, indices_r):
            y_t[i]=-y_t[i]
    
    Eout += Err(x_t, y_t, w) 
    
print('\nPara modelo lineal:\n')
print('Ein medio:', Ein/N)
print('Eout medio', Eout/N)


##########

input("\n--- Pulsar intro para continuar ---\n")
print('Apartado d (No lineal) \n')

#Reinicio Ein y Eout
Ein = 0
Eout = 0
for i in range(N):
    np.random.seed(i)
    pto_ini = np.zeros((6,1))
    #Hago la muestra
    muestra = simula_unif(tam, 2, 1)
    f_muestra = []
    for i in range(tam):
        f_muestra.append(f(muestra[:,0][i], muestra[:,1][i]))
    f_muestra = np.array(f_muestra)
    f_muestra = f_muestra.reshape(-1,1)
    
    #Añado ruido a la muestra
    f_ruido = f_muestra.copy()
    porcentaje = 0.1
    indices_r = np.random.choice(f_ruido.size, int(porcentaje*f_ruido.size), replace=False)
    for i in range(f_ruido.size):
        if np.isin(i, indices_r):
            f_ruido[i]=-f_ruido[i]
    
    #Sumo a Ein
    x = np.c_[np.ones((1000,1)), muestra, muestra[:,0]*muestra[:,1], muestra[:,0]*muestra[:,0], muestra[:,1]*muestra[:,1] ]
    y = f_ruido
    w = sgd(x, y, pto_ini, lr, maxIt, error2get, tam_batch)
    Ein += Err(x,y, w)
    
    test = simula_unif(tam, 2, 1)
    x_t =np.c_[np.ones((1000,1)), test, test[:,0]*test[:,1], test[:,0]*test[:,0], test[:,1]*test[:,1] ]
    y_t = []
    for i in range(tam):
        y_t.append(f(x_t[:,0][i], x_t[:,1][i]))
    y_t = np.array(y_t)
    y_t = y_t.reshape(-1,1)
    indices_r = np.random.choice(y_t.size, int(porcentaje*y_t.size), replace=False)
    for i in range(y_t.size):
        if np.isin(i, indices_r):
            y_t[i]=-y_t[i]
    Eout += Err(x_t, y_t, w) 
    
print('\nPara modelo no lineal:\n')
print('Ein medio:', Ein/N)
print('Eout medio', Eout/N)