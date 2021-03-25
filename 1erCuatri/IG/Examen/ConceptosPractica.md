#P4
##Textura
La imagen de textura se representa en un cuadrado [0,1]X[0,1]. Es decir que (0,0) es la esquina superior izqd, (1,1) la esquina inferior derecha, (0,1) esquina superior izqd, (1,0) esquina inferior derecha.

En la tabla de texturas, la entrada i tiene las cordenadas de textura asociadas al vertice i.

##Normales
Si me pide calcular normales, puedo llamar a inicializar, donde se calculan solos. Sino, puedo copiar el metodo que sale ahí.

##Material (en objeto nodografoescena)
Tengo un objeto de malla creado.
1. Creo una clase NodoObjeto : public NodoGrafoEscena.
2. Pongo nombre con PonerNombre(""); e id con ponerIdentificador(int);
3. Creo textura: Textura *text= new Textura(_ruta_);
4. Agrego el material: agregar(new Material(text, a, b, c, d));
5. Agrego el objeto: agregar(new ObjetoQTengo());

a-> reflectividad ambiental
b-> componente difusa
c-> componente especular
d-> exponente especular (debo mantenerlo sobre 20 y bajarlo para materiales solo difusos)

**Ejemplos**
difuso-especular->(0.3,0.4,0.4,20)
puro difuso->     (0.1,0.9,0.1,15)
puro espcular->   (0.1,0.1,0.9,25)

#P5

##Hacer un circulo de objetos, cada nodo con su id
````
for(int i = 0; i < n; i++){
   	float alpha = i*2.0*M_PI/n;

	agregar(MAT_Traslacion(n*cos(alpha),0,n*sin(alpha)));
	NodoCil * nodo=new NodoCil();

	nodo->ponerIdentificador(9000000+i);
	agregar(nodo);
  }
````

Este sería basicamente en codigo para crear un circulo de objetos, en este caso NodoCil. También se le pone un identificador a cada objeto del circulo, de manera que pueda acceder a el. Cada nodo (en el codigo de NodoCil) debería haber un nombre puesto pero no identificador. En caso de que quiera mover el centro del círculo, debería haber una clase por encima donde añadir el círculo, y luego agregarle una matriz de traslacion.
