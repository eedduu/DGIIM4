# -*- coding: utf-8 -*-
#Autor: Eduardo Morales Muñoz

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split 

#### Parte 1 

#Cargo el dataset iris
iris = datasets.load_iris()

#Selecciono la primera y la tercera columna
X = np.array(iris.data)
X1 = np.array(X[:, 0])
X2 = np.array(X[:, 2])

#Guardo las clases y las divido las tres clases en 3 arrays
Y = np.array(iris.target)
Y1= np.where(Y==0)
Y2= np.where(Y==1)
Y3= np.where(Y==2)

#Dibujo la primera columna, añado un scatter por cada clase 
#a representar, le asigno color y etiqueta
plt.scatter(X1[0:50], Y1, c="orange", label="0")
plt.scatter(X1[50:100], Y2, c="black",label="1")
plt.scatter(X1[100:150], Y3, c="green",label="2")

#Muestro la leyenda y el gráfico
plt.legend()
plt.show()

#Lo mismo de arriba pero con la tercera columna
plt.scatter(X1[0:50], Y1, c="orange", label="0")
plt.scatter(X2[50:100], Y2, c="black", label="1")
plt.scatter(X2[100:150], Y3, c="green", label="2")

plt.legend()
plt.show()

#### Parte 2

#Agrego una columna con la clase a los datos para no perder la información
X_2 = np.insert(X, X.shape[1], Y, 1)

#Divido los datos en entrenamiento y test de forma aleatoria
#con una proporción del 75% usando esta función de scikit-learn
train, test = train_test_split(X_2, test_size = 0.25)
    

####Parte 3
#Creo un array con 100 valores equiespaciados entre el 0 y 4pi (incluidos)
ar = np.arange(0, 4*np.pi+4*np.pi/99, 4*np.pi/99, float)

    

#Dibujo las graficas de las respectivas funciones en base al
#array anterior. Uno "o" para dibujarlas con puntos. Añado
#color y etiquetas
plt.plot(np.sin(ar), 'o', color="green", label="seno")

plt.plot(np.cos(ar), "o", color="black", label="coseno")

plt.plot(np.tanh(np.sin(ar)+np.cos(ar)), 'o', color="red", label="tanh")

plt.legend()
plt.show()