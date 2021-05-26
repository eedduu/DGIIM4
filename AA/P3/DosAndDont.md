# Modelos a usar
Puedo usar modelos lineales:
- PLA Pocket
- PLA
- Regresión lineal
- Regresión logística

# Que tengo que hacer
- **Análisis** del problema
- **Exploración** de los datos
- Formulación de **Hipótesis**
- **Entrenar** al modelo
- **Validar** los datos
- **Discutir** los resultados

# Problemas:
Hay uno de **clasificación** (discreto) sobre sensorless drive diagnosis. Son datos extraidos de las señales eléctricas del coche. Los datos muestran 11 clases distintas. Es decir que a partir de esas características tengo que seleccionar una etiqueta para el coche en cuestión (1-11).

## Cositas
Deberia dividir primero datos de training y test, y luego los de entrenamiento los normalizo. Con los datos de test no puedo hacer nada, tengo que aplicar la hipótesis (cross-validation, 5-fold o 10-fold). De validation cojo la mejor de los datos de entrenamiento, es decir solo puedo coger una hipotesis. No puedo ejecutar distintas hipótesis y coger la mejor.

````
scaler = StandardScaler()
train_X = scaler.fit_transform( train_X )
test_X = scaler.transform( test_X )
````
Outlayer es cuando está a 4 desviaciones típicas de la media
