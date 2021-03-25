# Ejercicio 2

Hay funciones C∞(I) que tienen unas derivadas sucesivas fáciles de calcular y que se pueden expresar mediante una fórmula más o menos asequible. De las funciones elementales casi todas, la más lata es precisamente ...la tan(x). Tiene algunas ventajas, por ejemplo, al ser impar, sus coeficientes pares en el desarrollo de Mc Laurin van a ser cero. A partir del tercer término se va complicando...buscad una expresión que os sea cómoda para las derivadas sucesivas....y suerte.

# Ejercicio 5

Imagino que te refieres a cuando intentas calcular el comportamiento en 0 de la función,efectivamente en cuanto comienzas a calcular el polinomio característico de la exponencial y del seno ,te das cuenta de que es justamente la expresión que aparece en la función y por tanto puedes aplicar la fórmula infinitesimal del resto El corolario 10.2.7 simplemente te dice que evalúes la derivada (n) de la función centrado en a en este caso 0 y lo dividas por el factorial de n Luego deberíamos derivar el seno 3 veces

```
f(x)=(-sen(x)) => f'(x)=(-cos(x)) => f''(x)=senx(x) => f'''(x) cos(x) y finalmente lo evalúas en 0 que es 1
Y el denominador es el factorial de 3
```

Luego tienes en positivo 1/3! (Creo que tu problema ha sido no empezar derivando con -sen(x) sino con sen(x) El resultado finalmente es 1/3! * 1/3! = 1/36 Echale un vistazo de todas formas al ejemplo 10.2.8 porque es prácticamente el mismo y toda la demostración se centra en ese ejercicio Espero que te haya ayudado

Veamos el límite en +∞, el otro lo puedes pensar tu (con cuidado porque ex funciona distinto en +∞ y en −∞

)

¿Cuánto vale limx→+∞x−sinxx

?

¿Cuánto vale limx→+∞ex−1−x−x2/2x5

? (este último lo puedes ver por L'H-B fácilmente, o pensando quien "puede" más en infinito, ¿un polinomio o la exponencial?).

Ahora haciendo cuentas ya tienes el limite del producto.

# Ejercicio 6

Hasta este último tema, la forma de ver si un punto critico era extremo relativo era de esta forma, se estudiaba el crecimiento de la función antes y después del punto critico, usando el signo de la derivada normalmente, y ya está, si hay cambio de signo hay extremo relativo. Este método es interesante cuando la expresión de las derivadas sucesivas de la función sea muy farragoso.

Ahora tenemos, caso de que las derivadas sucesivas sean "asequibles". un método alternativo. Además sólo necesito el valor de las sucesivas derivadas en el punto crítico. Pongamos un par de ejemplos.

1.- f(x)=x4, ∀x∈R . Evidentemente, no hace falta hacer cuentas, sabemos que esta función tiene en x=0 un mínimo relativo. Pero vamos a utilizar el resultado consecuencia de la FIR (fórmula infinitesimal del resto). Tenemos f(x)=x4, f′(x)=4x3⋯f′′′(x)=24x, f(iv(x)=24

.

Por tanto al evaluarla en el punto crítico x=0 obtenemos

f′(0)=⋯f′′′(0)=0, f(iv(0)=24>0

Como f(iv(0)>0 y es una derivada cuarta (par) todas las derivadas anteriores en el punto crítico son nulas, en ese punto, esto es, en x=0

tenemos un mínimo.

2.-f(x)=x5, ∀x∈R

. Repitamos el proceso anterior y obtenemos

f′(0)=⋯f(0(iv)=0, f(v(0)=5!>0

. Como ahora la primera derivada no nula es la quinta (impar) el punto crítico no es extremo relativo.

Creo que con estos ejemplos se ve mejor. El método con cualquier función que se deje derivar sucesivamente con una cierta facilidad.

# ejercicio 6c

Creo que lo primero que tenemos que ver es que la función f(x)=x2|x|e−|x|≥0 ∀x∈R y además f(0)=0

. Hala, ya tenemos mínimo absoluto que además, al estar en el interior del intervalo de definición, es relativo.

Si has leido el mensaje que he puesto a Isabel sobre la conveniencia o no de cada método para ver cuando un punto crítico es extremo relativo. En este caso te recomiendo que estudies, usando la primera derivada, el crecimiento de la función a ambos lados del punto crítico ya que las derivadas sucesivas no parecen muy directas.

La otra cosa que tienes que tener en cuenta es que la función es par. Estúdiala para 0≤x

, y luego completa por simetría. Verás que hay más extremos...

Házlo y me lo cuentas. Intenta dibujarla...

Creo que con esto sale ya ¿no?

Hay una cosa que no habéis calculado bien. La función que estamos estudiando tiene derivadas en cero hasta orden 2, pero NO tiene derivada de orden 3 en cero ya que limx→0+f′′′(x)=6limx→0−f′′′(x)=−6 Así no puedo aplicar el resultado y no puedo decidir ya que todas las derivadas que puedo calcular son cero. No hay por tanto contradicción entre los dos métodos, uno dice que hay mínimo y el otro no cumple todas las hipótesis del teorema por lo que no dice nada.

# Ejercicio 8

Vimos ayer, 15-4, que las condiciones de este ejercicio implican en particular, usando recurrencia, que la función es C∞(I). Sólo quedaba ver que la función coincidia con su desarrollo de Taylor usando una cota uniforme para todas las derivadas, eso es posible teniendo en cuenta que basta acotar la función y sus dos primeras derivadas, ya que las demás son esencialmente iguales a éstas. Como, por recurrencia, fn(a)=0, ∀n∈N. Ésto probaria que f(x)=0, ∀x∈I

.

Bien, os prometí otra demostración sin usar Taylor. Sólo necesitamos saber que si una función en un intevalo es igual a su derivada, básicamente, es la exponencial, hecho que ya hemos probado, es una consecuencia del TVM. Vamos a ello. Tenemos una f:I⟶R; f′′(x)=f(x),∀x∈I . Sabemos también que ∃a∈I;f(a)=f′(a)=0

. Definimos la función

h(x)=f(x)−f′(x)

tenemos entonces que h′(x)=f′(x)−f(x)=−h(x) para todo punto de I. Como (h(x)ex)′=h′(x)ex+h(x)ex=0, deducimos que h(x)ex=C,∀x∈I, para C una constante. Luego h(x)=Ce−x,∀x∈I, como h(a)=0⇒C=0, luego h(x)=0⇒f(x)=f′(x),∀x∈I. Mediante un razonamiento parecido al anterior probaríamos que f(x)=Kex,∀x∈I, con K una constante real. Como f(a)=0⇒K=0 y por ende f(x)=0,∀x∈I

.

Como veis, hemos probado lo mismo sin necesidad de Taylor, sólo con argumentos básicos del tema de derivadas.

¿Cómo lo veis?

# Ejercicio 9

He reflexionado sobre lo que hablamos ayer y creo que con estas ideas esta más claro.

Partimos del desarrollo del coseno. Tenemos en cuenta dos cosas, una que, al ser el coseno una función par, su polinomio de Mc Laurin esta formado por potencias pares (los coeficientes impares se anulan). Así el polinomio de grado dos y tres son el mismo.

Si uso el Teorema de la fórmula de Taylor (Thrm 10.2.15) para el polinomio de grado 2 obtengo

cosx−1+x22=sincx33! , para algún cierto c∈(0,π). La expresión de la derecha es mayor o igual que cero ya que el seno en ese intervalo lo es, por tanto cosx−1+x22≥0

.

Si ahora hago lo análogo para el polinomio de grado tres obtengo

cosx−1+x22=cosdx44! , para algún cierto d∈(0,π)

. Como el coseno es siempre menor o igual que uno ya tengo la otra desigualdad.

Fijaos que, una vez que tenemos claro el desarrollo de Taylor, y el cómputo del error, la acotación es "casi" inmediata.

¿Ahora se ve mejor?

Cuidaros... ;-)
