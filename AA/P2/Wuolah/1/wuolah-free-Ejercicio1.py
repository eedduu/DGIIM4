# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Angeles Caballero Floro
"""

import numpy as np
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(1)

print('\n---------------------------- LA COMPLEJIDAD DE H Y EL RUIDO ------------------------------\n')
#---------------------------------------------------------------------------------#
#-----------Funciones implementadas por los profesores de prácticas---------------#
#---------------------------------------------------------------------------------#


# Función que nos permite simular los elementos de un conjunto de datos con 
# distribución uniforme

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))


# Función que nos permite simular los elementos de un conjunto de datos con 
# distribución gaussiana
    
def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


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


# Función que devuelve el signo (+ o -) de un dato 
# La funcion np.sign(0) da 0, lo que nos puede dar problemas
    
def signo(x):
	if x >= 0:
		return 1
	return -1


# Función que facilita la clasificación (etiquetas) de los elementos de un conjunto
# de datos
    
def f(x, y, a, b):
	return signo(y - a*x - b)



# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
print('\n------------------------------------------------------------------------------------------')
print('Ejercicio 1:')
print('------------------------------------------------------------------------------------------')

#---------------------------------------------------------------------------------#
#------------------------------Generación de datos--------------------------------#
#---------------------------------------------------------------------------------#

x = simula_unif(50, 2, [-50,50])
print("Conjunto X con distribución uniforme, generado satisfactoriamente...")
x2 = simula_gaus(50, 2, np.array([5,7]))
print("Conjunto X con distribución gausiana, generado satisfactoriamente...")

#---------------------------------------------------------------------------------#
#-------------------------Empleo de los datos generados---------------------------#
#---------------------------------------------------------------------------------#

# Plot outputs X
plt.figure(1)
plt.scatter(x[:,0], x[:,1]) #Para puntos
plt.title('EJERCICIO 1.1 A - GRAFICA DE PUNTOS CON SIMULA_UNIF')
plt.show()  #Muestra una grafica de puntos aleatorios con distribucion uniforme

# Plot outputs x2
plt.figure(2)
plt.scatter(x2[:,0], x2[:,1]) #Para puntos
plt.title('EJERCICIO 1.1 B - GRAFICA DE PUNTOS CON SIMULA_GAUS')
plt.show()  #Muestra una grafica de puntos aleatorios con distribucion gausiana

input("\n--- Pulsar tecla para continuar ---\n")

###################################################################################
###################################################################################
###################################################################################

# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

#---------------------------------------------------------------------------------#
#------------------------------Generación de datos--------------------------------#
#---------------------------------------------------------------------------------#

# Calculamos X
X = simula_unif(50, 2, [-50,50])

print("Conjunto X de 50 elementos, generado satisfactoriamente...")

#....Generamos una recta caracterizada por los valores a (pendiente) y b (término
#....independiente) 
a, b = simula_recta([-50,50])

# Calculamos Y
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

#---------------------------------------------------------------------------------#
#-------------------------Empleo de los datos generados---------------------------#
#---------------------------------------------------------------------------------#

# Plot outputs sin etiqueta
plt.figure(3)
plt.scatter(X[:, 0], X[:, 1]) #Para puntos
plt.title('EJERCICIO 1.2 A - GRAFICA SIN ETIQUETAR')
plt.show() 

# Plot outputs con etiqueta
lineaX = np.linspace(-50, 50, Y.size)
lineaY = a * lineaX + b

#.... Mostramos la gráfica por pantalla y el error
plt.figure(4)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.plot(lineaX, lineaY, 'r-', linewidth=2)
plt.title('EJERCICIO 1.2 A - GRAFICA DE PUNTOS CON ETIQUETAS (SEPARACION)')
plt.show()
plt.pause(0.01)


input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la
# recta usada para ello

# Array con 10% de indices aleatorios para introducir ruido


#---------------------------------------------------------------------------------#
#-------------------------Funciones implementadas por mi--------------------------#
#---------------------------------------------------------------------------------#

def ruido (Y,porcentaje):
    # Separamos los elementos que son positivos de los negativos y guardamos sus
    # indices en vectores separados
    Y_pos = np.where(Y > 0)
    Y_neg = np.where(Y < 0)
    
    Y_pos = np.array(Y_pos)
    Y_neg = np.array(Y_neg)
    
    # Seleccionamos una muestra del tamaño especificado (10%)
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
        
print("Ruido introducido en la muestra de forma adecuada...")

# Con esto obtenemos una nueva variable Y en la que hemos simulado un 10% de ruido


#---------------------------------------------------------------------------------#
#-------------------------Empleo de los datos generados---------------------------#
#---------------------------------------------------------------------------------#

# Plot outputs
lineaX = np.linspace(-50, 50, Y.size)
lineaY = a * lineaX + b
#.... Mostramos la gráfica por pantalla y el error
plt.figure(5)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.plot(lineaX, lineaY, 'r-', linewidth=2)
plt.title('EJERCICIO 1.2 B - GRAFICA DE PUNTOS CON ETIQUETAS (SEPARACION) CON RUIDO')
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

####################################################################################
####################################################################################
####################################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de
# clasificación de los puntos de la muestra en lugar de una recta

#---------------------------------------------------------------------------------#
#-----------Funciones implementadas por los profesores de prácticas---------------#
#---------------------------------------------------------------------------------#

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    # Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy - min_xy) * 0.01

    # Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0] - border_xy[0]:max_xy[0] + border_xy[0] + 0.001:border_xy[0],
             min_xy[1] - border_xy[1]:max_xy[1] + border_xy[1] + 0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)

    # Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu', vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2,
               cmap="RdYlBu", edgecolor='white')

    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)), X.shape[0]),
                         np.linspace(round(min(min_xy)), round(max(max_xy)), X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX, YY, fz(positions.T).reshape(X.shape[0], X.shape[0]), [0], colors='black')

    ax.set(
        xlim=(min_xy[0] - border_xy[0], max_xy[0] + border_xy[0]),
        ylim=(min_xy[1] - border_xy[1], max_xy[1] + border_xy[1]),
        xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    
#---------------------------------------------------------------------------------#
#---------------------------------------F1----------------------------------------#
#---------------------------------------------------------------------------------#

def f1(x):
    return (x[:, 0]-10)**2 + (x[:, 1]-20)**2 - 400

# Dado el conjunto de datos X (calculado en el ejercicio 2) lo etiquetamos segun la
# primera función que se nos da en el pdf (f1)
Y = f1(X)
print("Etiquetando segun f1...")

#Añadimos ruido a nuestras etiquetas
Y_pos, Y_neg, index_pos, index_neg = ruido (Y, 0.1)

for i in range(Y.size):
    if np.isin(i,Y_pos[0][index_pos]):
        Y[i] = -Y[i]
    
for i in range(Y.size):
    if np.isin(i,Y_neg[0][index_neg]):
        Y[i] = -Y[i]

print("Ruido introducido en la muestra de forma adecuada...")

# Plot outputs      
plot_datos_cuad(X, Y, f1)

input("\n--- Pulsar tecla para continuar ---\n")

#---------------------------------------------------------------------------------#
#---------------------------------------F2----------------------------------------#
#---------------------------------------------------------------------------------#

def f2(x):
    return 0.5*(x[:, 0]+10)**2 + (x[:, 1]-20)**2 - 400

# Dado el conjunto de datos X (calculado en el ejercicio 2) lo etiquetamos segun la
# segunda función que se nos da en el pdf (f2)
Y = f2(X)
print("Etiquetando segun f2...")

#Añadimos ruido a nuestras etiquetas
Y_pos, Y_neg, index_pos, index_neg = ruido (Y, 0.1)

for i in range(Y.size):
    if np.isin(i,Y_pos[0][index_pos]):
        Y[i] = -Y[i]
    
for i in range(Y.size):
    if np.isin(i,Y_neg[0][index_neg]):
        Y[i] = -Y[i]
        
print("Ruido introducido en la muestra de forma adecuada...")

# Plot outputs        
plot_datos_cuad(X, Y, f2)

input("\n--- Pulsar tecla para continuar ---\n")


#---------------------------------------------------------------------------------#
#---------------------------------------F3----------------------------------------#
#---------------------------------------------------------------------------------#

def f3(x):
    return 0.5*(x[:, 0]-10)**2 + (x[:, 1]+20)**2 - 400

# Dado el conjunto de datos X (calculado en el ejercicio 2) lo etiquetamos segun la
# tercera función que se nos da en el pdf (f3)
Y = f3(X)
print("Etiquetando segun f3...")

#Añadimos ruido a nuestras etiquetas
Y_pos, Y_neg, index_pos, index_neg = ruido (Y, 0.1)

for i in range(Y.size):
    if np.isin(i,Y_pos[0][index_pos]):
        Y[i] = -Y[i]
    
for i in range(Y.size):
    if np.isin(i,Y_neg[0][index_neg]):
        Y[i] = -Y[i]

print("Ruido introducido en la muestra de forma adecuada...")

# Plot outputs       
plot_datos_cuad(X, Y, f3)

input("\n--- Pulsar tecla para continuar ---\n")


#---------------------------------------------------------------------------------#
#---------------------------------------F4----------------------------------------#
#---------------------------------------------------------------------------------#

def f4(x):
    return x[:, 1] - 20*(x[:, 0]**2) - 5*x[:, 0] +3

# Dado el conjunto de datos X (calculado en el ejercicio 2) lo etiquetamos segun la
# ultima función que se nos da en el pdf (f4)
Y = f4(X)
print("Etiquetando segun f4...")

# En este caso no podemos meter ruido, ya que todos los valores son del mismo "tipo"
# NEGATIVOS

# Plot outputs        
plot_datos_cuad(X, Y, f4)


#---------------------------------------------------------------------------------#