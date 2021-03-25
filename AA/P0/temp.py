# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#Parte 1 

iris = datasets.load_iris()
X = iris.data
Y = iris.target
X2= X[1:3:2] 


####Parte 3
arr = []
for i in range (0, 100):
    a = i*4*np.pi/99
    arr.append(a)
    

dibujo1 = plt.plot(np.sin(arr), 'o', color="green")
#plt.show()

dibujo2 = plt.plot(np.cos(arr), "o", color="black")
#plt.show()


#tanhiper(x) = np.tanh(np.sin(x)+np.cos(x))
dibujo3 = plt.plot(np.tanh(np.sin(arr)+np.cos(arr)), 'o', color="red")