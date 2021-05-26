# Clasificacion (sensor drive noseque)

## Descripcion del problema
En el problema de clasificación tenemos un dataset de características sobre la corriente de un motor. Los y las etiquetas se proporcionan juntas, siendo las etiquetas la última columna de los datos proporcionados. Los elementos del problema son los siguientes:

- **X**: diferentes características de la corriente del motor. En concreto son 48 características, por lo que un elemento de X será un vector de 48 elementos. El motor tiene elementos intactos y otros defectuosos. Se usó el método de descomposición empírico para generar los datos. $X = \mathbb{R}^{48} , x = (x_{0}, ..., x_{47}), x \in X$
- **Y**: 11 clases diferentes de motor, representadas por un entero. $Y = \{1, 2, 3, ..., 11\}$
- **f**: función que, en base a las características anteriores ($x \in X$) de un motor le asigna una etiqueta en función de su tipo. $f: X \rightarrow Y ; f(x)=y , x \in X, y \in Y$


# Regresión

## Descripcion del problema
El problema de regresión se compone de un dataset ofrecido en dos ficheros. En uno tenemos la temperatura crítica superconductora (en º Kelvin) del elemento junto a su fórmula química, mientras que en el otro fichero tenemos las características de cada elemento químico extraídas a partir de su fórmula química. Los elementos del problema son:

- **X**: características del elemento en función de su fórmula química. En concreto son 81 características, por lo que un elemento de X será un vector de 81 elementos, donde el primero es un natural (número de elementos del elemento químico) y el resto son valores reales positivos. $X = \mathbb{N} \times \mathbb{R^{+}}^{80} ; x = (x_0, ..., x_{80}), x \in X$

- **Y**: temperatura crítica superconductora del elemento en cuestión, en grados Kelvin. Por tanto se trata de un real positivo (se entiende que no se alcanzará el 0 absoluto). $Y = \mathbb{R}^{+}$
- **f**: función que, dadas las características de un elemento superconductor, le asigna su temperatura crítica superconductora. $f: X \rightarrow Y ; f(x)=y , x \in X, y \in Y$
