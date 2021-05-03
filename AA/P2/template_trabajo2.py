# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante:  Eduardo Morales Muñoz
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


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


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

x = simula_unif(50, 2, [-50,50])

#Dibujo la gráfica
plt.scatter(x[:,0],x[:,1])
plt.title("Ej 1.1-Simula uniforme")
plt.show()

x = simula_gaus(50, 2, np.array([5,7]))

#Dibujo la gráfica
plt.scatter(x[:,0],x[:,1])
plt.title("Ej 1.1-Simula gauss")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente



# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)


#Nube de puntos N=100
x = simula_unif(100,2,[-50,50])

#Dibujo la gráfica
plt.scatter(x[:,0],x[:,1])
plt.title("Ej1.2a-Grafica sin etiquetas")
plt.show()

a, b = simula_recta([-50,50]) #simulo la recta

#Asigno etiquetas
etiq = []
for i in range(100):
    etiq.append(f(x[:,0][i], x[:,1][i], a, b))
Y = np.array(etiq)

#Dibujo la grafica con etiquetas y recta
plt.scatter(x[:,0],x[:,1], c=Y)

lineaX = np.linspace(-50, 50, Y.size)
lineaY = a * lineaX + b

plt.plot(lineaX, lineaY, 'r-')
plt.title("Ej1.2a-Grafica con etiquetas y recta")

plt.show()



input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido
def etiquetar_ruido(Y, porcentaje):
    #Selecciono elementos negativos y positivos del array de etiquetas y los meto en un array
    pos = np.where(Y > 0)
    neg = np.where(Y < 0)
    
    #Convierto el array a un array de numpy
    pos = np.array(pos)
    neg = np.array(neg)
    
    #Selecciono un 10% aleatorio de los indices de cada array (positivo y negativo)
    index_p = np.random.choice(pos.size, int(porcentaje*pos.size), replace=False)
    index_n = np.random.choice(neg.size, int(porcentaje*neg.size), replace=False)
    
    return pos, neg, index_p, index_n 

pos, neg, ind_p, ind_n = etiquetar_ruido(Y, 0.1)

#Añado el ruido a las etiquetas con los arrays que he obtenido
for i in range(Y.size):
    if np.isin(i,pos[0][ind_p]):
        Y[i]= -Y[i]
for i in range(Y.size):
    if np.isin(i, neg[0][ind_n]):
        Y[i]= -Y[i]

#Dibujo la grafica con ruido, etiquetas y recta
plt.scatter(x[:,0],x[:,1], c=Y)

lineaX = np.linspace(-50, 50, Y.size)
lineaY = a * lineaX + b

plt.plot(lineaX, lineaY, 'r-')
plt.title("Ej1.2b-Grafica con ruido")

plt.show()



input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    
#Defino la funcion

def f1(x):
    return (x[:,0]-10)**2 + (x[:,1]-20)**2 - 400

#Defino la funcion
def f2(x):
    return 0.5*(x[:,0]+10)**2 + (x[:,1]-20)**2 - 400

#Defino la funcion
def f3(x):
    return 0.5*(x[:,0]-10)**2 - (x[:,1]+20)**2 - 400

#Defino la funcion
def f4(x):
    return x[:,1] - 20*x[:,0]**2 - 5*x[:,0] + 3

#Pinto las graficas sobre las etiquetas del apartado b

plot_datos_cuad(x, Y, f1, "Grafica de f1")
plot_datos_cuad(x, Y, f2, "Gráfica de f2")
plot_datos_cuad(x, Y, f3, "Gráfica de f3")
plot_datos_cuad(x, Y, f4, "Gráfica de f4")


# Ahora reasigno las etiquetas en base a la función que uso para pintar.

#Asigno las etiquetas
Yaux = f1(x)
for i in range(Y.size):
    Y[i]=signo(Yaux[i])


#Añado ruido
pos, neg, ind_p, ind_n = etiquetar_ruido(Y, 0.1)

for i in range(Y.size):
    if np.isin(i,pos[0][ind_p]):
        Y[i]= -Y[i]
for i in range(Y.size):
    if np.isin(i, neg[0][ind_n]):
        Y[i]= -Y[i]
        
#Dibujo la gráfica con la función dada
plot_datos_cuad(x, Y, f1, "Grafica de f1 etiquetas modificadas")




#Asigno las etiquetas
Yaux = f2(x)
for i in range(Y.size):
    Y[i]=signo(Yaux[i])

#Añado ruido
pos, neg, ind_p, ind_n = etiquetar_ruido(Y, 0.1)

for i in range(Y.size):
    if np.isin(i,pos[0][ind_p]):
        Y[i]= -Y[i]
for i in range(Y.size):
    if np.isin(i, neg[0][ind_n]):
        Y[i]= -Y[i]
        
#Dibujo la gráfica con la función dada
plot_datos_cuad(x, Y, f2, "Gráfica de f2 etiquetas modificadas")




#Asigno las etiquetas
Yaux = f3(x)
for i in range(Y.size):
    Y[i]=signo(Yaux[i])

#Añado ruido
pos, neg, ind_p, ind_n = etiquetar_ruido(Y, 0.1)

for i in range(Y.size):
    if np.isin(i,pos[0][ind_p]):
        Y[i]= -Y[i]
for i in range(Y.size):
    if np.isin(i, neg[0][ind_n]):
        Y[i]= -Y[i]

        
#Dibujo la gráfica con la función dada
plot_datos_cuad(x, Y, f3, "Gráfica de f3 etiquetas modificadas")




#Asigno las etiquetas
Yaux = f4(x)
for i in range(Y.size):
    Y[i]=signo(Yaux[i])

#Añado ruido
pos, neg, ind_p, ind_n = etiquetar_ruido(Y, 0.1)

for i in range(Y.size):
    if np.isin(i,pos[0][ind_p]):
        Y[i]= -Y[i]
for i in range(Y.size):
    if np.isin(i, neg[0][ind_n]):
        Y[i]= -Y[i]
        

#Dibujo la gráfica con la función dada
plot_datos_cuad(x, Y, f4, "Gráfica de f4 etiquetas modificadas")

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(x, label, max_iter, vini):
    cambio = True
    w = vini
    j=0
    while cambio and j<max_iter:
        j+=1
        cambio = False
        for i in range(x.shape[0]):
            if signo(w.T.dot(x[i])) != label[i]:
                w = w + label[i]*x[i]
                cambio = True
    
    
    return w, j

def errorPLA(x, y, w):
    error=0
    for i in range(y.size):
        error += max(0, -y[i]*w.dot(x[i]))
    return np.mean(error)

#DATOS DEL EJERCICIO 1.2A
#añado columna de 1's a los datos del ejercicio 1.2a

print("Ejercicio 2.1a\n")
x = np.insert(x, 0, np.ones(100), 1)

#inicializo w
p_ini = np.zeros(3)

#Asigno labels
etiq = []
for i in range(100):
    etiq.append(f(x[:,1][i], x[:,2][i], a, b))
Y = np.array(etiq)

print("Ejecutando PLA con punto inicial vector de 0")
w, iteraciones = ajusta_PLA(x, Y, 10000, p_ini)

print("Número de iteraciones necesarias para converger: ", iteraciones)
print("Ein: ", errorPLA(x, Y, w))

sgdLR_x = np.linspace(-50, 50, Y.size)
sgdLR_y = (-w[0] - w[1]*sgdLR_x) / w[2]
plt.title('Ajuste PLA (datos sin ruido)')
plt.scatter(x[:,1], x[:,2], c=Y)
plt.plot(sgdLR_x, sgdLR_y, 'r-')
plt.show()

# Random initializations
iterations = []
vectores = []
Ein = []
for i in range(10):
    p_ini = np.random.rand(3)
    vectores.append(p_ini)
    w, iteraciones = ajusta_PLA(x, Y, 1000, p_ini)
    iterations.append(iteraciones)
    Ein.append(errorPLA(x, Y, w))


print("\nEjecutando PLA con punto inicial vector aleatorio entre 0 y 1")
    
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

print("Ein medio: {}".format(np.mean(np.asarray(Ein))) )

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b
print("Ejercicio 2.1b\n")
#ya tengo x del ejercicio anterior

#inicializo w
p_ini = np.zeros(3)

#Asigno labels
etiq = []
for i in range(100):
    etiq.append(f(x[:,1][i], x[:,2][i], a, b))
Y = np.array(etiq)

pos, neg, ind_p, ind_n = etiquetar_ruido(Y, 0.1)

#Añado el ruido a las etiquetas con los arrays que he obtenido
for i in range(Y.size):
    if np.isin(i,pos[0][ind_p]):
        Y[i]= -Y[i]
for i in range(Y.size):
    if np.isin(i, neg[0][ind_n]):
        Y[i]= -Y[i]
        

print("Ejecutando PLA con punto inicial vector de 0")
w, iteraciones = ajusta_PLA(x, Y, 2000, p_ini)


print("Número de iteraciones necesarias para converger: ", iteraciones)
print("Ein: ", errorPLA(x, Y, w))


sgdLR_x = np.linspace(-50, 50, Y.size)
sgdLR_y = (-w[0] - w[1]*sgdLR_x) / w[2]
plt.title('Ajuste PLA (datos con ruido)')
plt.scatter(x[:,1], x[:,2], c=Y)
plt.plot(sgdLR_x, sgdLR_y, 'r-')
plt.show()

# Random initializations
iterations = []
vectores = []
Ein = []
for i in range(0,10):
    p_ini = np.random.rand(3)
    vectores.append(p_ini)
    w, iteraciones = ajusta_PLA(x, Y, 2000, p_ini)
    iterations.append(iteraciones)
    Ein.append(errorPLA(x, Y, w))


print("\nEjecutando PLA con punto inicial vector aleatorio entre 0 y 1")
    
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

print("Ein medio: {}".format(np.mean(np.asarray(Ein))) )


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def gradE(x, y, w):
    return -(y*x)/(1+np.exp(y*w.dot(x)))

def Error(x, y, w):
    suma = np.log(1+np.exp(-y*x.dot(w)))
    return np.mean(suma)


def sgdRL(x, y, lr, w, error2get):
    w_old = w.copy()
    w_new = w.copy()
    #Hago una epoca fuera del bucle para que haya diferencia entre los w de una epoca y otra. Primero hago las permutaciones aleatorias de los elementos
    ind = np.random.permutation(y.size)
    x = x[ind, :]
    y = y[ind]   
    for i in range(y.size):
        w_new = w_new - lr*gradE(x[i],y[i],w_new)
    epocas = 1
    while np.linalg.norm(w_old-w_new)>error2get:
        ind = np.random.permutation(y.size)
        x = x[ind, :]
        y = y[ind] 
        w_old = w_new.copy()
        epocas+=1
        for i in range(y.size):
            w_new = w_new - lr*gradE(x[i],y[i],w_new)
        
    return w_new, epocas




#Funcion similar a la dada, que calcula una recta pero a partir de dos puntos de un conjunto dado
def simula_recta_cjto(X):
    index = np.random.choice(X.shape[0], 2, replace= False)
    points = X[index]
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b
    
epocas = []
Ein = []
Eout = []


for i in range(2):
    #Cojo 1200 datos de [0,2]x[0,2] de forma uniforme
    X = simula_unif(1200, 2, [0,2])
    
    
    #Obtengo una recta a partir de dos puntos del conjunto
    a, b = simula_recta_cjto(X)
    
    #Evalúo los puntos según la recta que he obtenido
    etiq = []
    for punto in X:
        f = punto[1] - a*punto[0] - b
        if (1/(1 + np.exp(f))) >= 0.5:
            etiq.append(1)
        else:
            etiq.append(-1)
            
    Y = np.array(etiq, dtype=np.float64)
    
    #Selecciono 100 muestras de las 1200 para usarlas en el entrenamiento, el resto los uso en el test
    index = np.random.choice(Y.size, 100, False)
    
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(Y.size):
        if np.isin(i,index):
            x_train.append(X[i])
            y_train.append(Y[i])
        else:
            x_test.append(X[i])
            y_test.append(Y[i])
            
    x_train = np.array(x_train, np.float64)
    y_train = np.array(y_train, np.float64)
    x_test = np.array(x_test, np.float64)
    y_test = np.array(y_test, np.float64)
    
    
    #Añado una columna de 1's a X
    x_train = np.insert(x_train, 0, np.ones(100), 1)
    x_train = np.array(x_train, dtype=np.float64)
    
    x_test = np.insert(x_test, 0, np.ones(1100), 1)
    x_test = np.array(x_test, dtype=np.float64)
    
    #Inicializo el vector de pesos con valores a 0
    w = np.zeros(3, dtype=np.float64)
    
    #Parametros
    lr=0.01
    error=0.01
    
    
    w, epoc = sgdRL(x_train.copy(), y_train.copy(), lr, w, error)
    epocas.append(epoc)
    In = Error(x_train, y_train, w)
    Ein.append(In)
    Errorout = Error(x_test, y_test, w)
    Eout.append(Errorout)
    


   
    
    
print('Valor medio de epocas que tarda el algoritmo: {}'.format(np.mean(np.asarray(epocas))))
print('Ein medio: {}'.format(np.mean(np.asarray(Ein))))
print('Eout medio: {}'.format(np.mean(np.asarray(Eout))))

##Ejecuto para hacer ejemplo

#Muestro las gráficas
recta_x = np.linspace(0, 2, y_train.size)
recta_y = (a*recta_x + b) 
plt.title('Ejemplo de Datos de entrenamiento etiquetados')
plt.scatter(x_train[:,1], x_train[:,2], c=y_train)
plt.xlim(0,2)
plt.ylim(0,2)
plt.plot(recta_x, recta_y, 'r-')
plt.show()


sgdLR_x = np.linspace(0, 2, y_train.size)
sgdLR_y = (-w[0] - w[1]*sgdLR_x) / w[2]
plt.title('Ejemplo de Conjunto de entrenamiento clasificado con el modelo')
plt.scatter(x_train[:,1], x_train[:,2], c=y_train)
#plt.xlim(0,2)
#plt.ylim(0,2)
plt.plot(sgdLR_x, sgdLR_y, 'r-')
plt.show()


sgdLR_x = np.linspace(0, 2, y_test.size)
sgdLR_y = (-w[0] - w[1]*sgdLR_x) / w[2]
plt.title('Ejemplo de conjunto de test clasificado con el modelo')
plt.scatter(x_test[:,1], x_test[:,2], c=y_test)
#plt.xlim(0,2)
#plt.ylim(0,2)
plt.plot(sgdLR_x, sgdLR_y, 'r-')
plt.show()
    






input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos

label4 = -1
label8 = 1

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


def grad(x, y, w):
    y = y.reshape(-1,1)
    suma = 2 * (x.dot(w)-y)*x
    media = np.mean(suma, axis =0)
    media = media.reshape(-1,1)
    return media

def sgd(x, y, lr, w, error2get, maxiter):
    iteraciones = 0
    while Err(x, y, w)> error2get and iteraciones < maxiter:
        iteraciones += 1
        w = w - lr*grad(x, y, w)
        
    return w, iteraciones

def ajusta_PLAP(x, label, max_iter, vini):
    cambio = True
    w = vini.copy()
    mejor_w= vini.copy()
    j=0
    while cambio and j<max_iter:
        j+=1
        cambio = False
        for i in range(x.shape[0]):
            if (np.dot(np.transpose(w), x[i]) * label[i])[0] <= 0:
                w = w + label[i]*x[i]
                cambio = True
                if Err(x, label, w) < Err(x, label,mejor_w):
                    mejor_w = w    
    return mejor_w, j




def Err(x,y,w):
    y = y.reshape(-1,1)
    suma = (x.dot(w)-y)**2
    media = np.mean(suma, axis = 0)
    media = media.reshape(-1,1)
    return float(media[0])


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 
lr = 0.01
error = 1e-14
w = np.zeros((3,1))
maxit = 40000
w_ini, iteraciones = sgd(x, y, lr, w, error, maxit)
Ein = Err(x,y,w_ini)
Eout = Err(x_test, y_test, w_ini)

print("Ein sin mejora: ", Ein)
print("Etest sin mejora: ", Eout)


#Muestro la gráfica sin mejora
sgdLR2_x = np.linspace(0, 1, y.size)
sgdLR2_y = (-w_ini[0] - w_ini[1]*sgdLR2_x) / w_ini[2]

fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Resultado Regresion Lineal (Entrenamiento)')
ax.set_xlim((0, 1))
ax.set_ylim((-8,0))
plt.plot(sgdLR2_x, sgdLR2_y, 'r-', label="Funcion estimada sin mejora")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Resultado Regresion Lineal (Test)')
ax.set_xlim((0, 1))
ax.set_ylim((-8,0))
plt.plot(sgdLR2_x, sgdLR2_y, 'r-', label="Funcion estimada sin mejora")
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

Maxiter= 100

w, epocas= ajusta_PLAP(x, y, Maxiter, w_ini)

Ein = Err(x,y,w)
Eout = Err(x_test, y_test, w)

print("Ein con mejora: ", Ein)
print("Etest con mejora: ", Eout)


#Muestr la grafica con mejora
sgdLR2_x = np.linspace(0, 1, y.size)
sgdLR2_y = (-w[0] - w[1]*sgdLR2_x) / w[2]

fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Resultado Mejora PLA (Entrenamiento)')
ax.set_xlim((0, 1))
ax.set_ylim((-8,0))
plt.plot(sgdLR2_x, sgdLR2_y, 'r-', label="Funcion estimada con mejora")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Resultado Mejora PLA (Test)')
ax.set_xlim((0, 1))
ax.set_ylim((-8,0))
plt.plot(sgdLR2_x, sgdLR2_y, 'r-', label="Funcion estimada con mejora")
plt.legend()
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

dvc = 3

cota1 = Ein + np.sqrt((8/y.size)*np.log((4*(((2*y.size)**dvc)+1))/0.05))
cota2 = Eout + np.sqrt((1/(2*y_test.size))*np.log(2/0.05))

print("Cota usando Ein: ", cota1)
print("Cota usando Etest", cota2)