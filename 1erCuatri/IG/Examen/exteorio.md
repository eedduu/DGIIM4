#Espacio afin

1. Demostrar por las propiedades del prod escalar que a=R(x,y,z), b=R(r,s,t), a*b=xr+ys+tz -> ej 7 pag 11

2. Igual prod vectorial -> ej 8 pag 11

3. Demostrar a·b=||a||·||b||·cos(alfa) ->ej 9 pag 12

#Modelado de objetos

1. Calcula el espacio que ocupa una representacion (da igual figura) segun vertices, por enumeracion espacial y modelado de fronteras (malla indexada) -> ej 10 pag 15

2. Calcular espacio que ocupa una malla indexada con topologia de rejilla ej 11 pag 15 (el 12 es parecido)

3. En una malla cerrada conexa, demostrar que el nº de vértices determina el nº de aristas y el nº de caras (y viceversa) ej 13 pag 17

4. Area total de una malla indexada de triangulos -> Ej 15 par 19 (resueltos pag 8)

5. Aristas aisladas -> apuntes wuolah tema2

6. Calcular normales -> ej 17 resuelto relacion pag 9

7. Copia de malla de triangulos, desplazada en la direccion de las normales. -> ej 18 resuelto relacion pag 9

8. Producto escalar invariante por rotacion ej 20 pag 20

9. En 2D la matriz de rotacion por un angulo tiene como inversa la matriz de rotacion por el angulo en negativo (3D tmb) ej 21 pag 21

10. las rotaciones no modifican la longitud del vector -> ej 22 pag 22

11. Grafo PHIGS -> ej 28 pag 27

#Visualizacion
1. Proyeccion perspectiva, view-frustum -> ej 35 pag 33

2. Demostrar que una recta en coordenadas de mundo se transforma en una recta de coordenadas de dispositivo -> ej 36 pag 34

3. Pasar ecuaciones implicitas en CC a otro tipo de proyección. -> ej 37 pag 35

4. Matriz de vista V y proyeccion P para proyeccion ortográfica -> ej 38 pag 37

5. Lo mismo para proyecion perspectiva -> ej 39 pag 38 (40 parecido)

6. Añadido de field of view -> ej 41 pag 41 (el 42 parecido, resuelto en relacion pag 14.

7. Asignar texturas a malla indexada -> ej 43 pag 42 resuelto relacion pag 15

8. Asignar texturas con sombreado de Gouroud o Phong (normales bien calculadas) -> ej 44 pag 43 resuelto relacion pag 16

9. Asignar texturas a un cubo ej 45 pag 43 resuelto pag 17

10. Iluminacion luz puntual, observador en un punto, calcular brillo max. -> ej 46 pag 43 resuelto pag 18

##Notas
- Cada vertice, k, son 3·4 bytes (4 tamaño usual de float/int)
- La enumeración espacial requiere k^3 bits. (1/8)k^3 bytes
- La representacion mediante mallas son k^2 vertices y 2k^2 caras. -> 36k^2 bytes (suponiengo float/int 4 bytes)
- Una malla topología de rejilla tiene (n+1)(m+1) vertices. Para saber lo que ocupa tengo q multiplicarlo por lo que ocupa un float/int (4 bytes normalmente) y por 3 (3 float por vertice) y sumarle el numero de triangulos por lo que ocupa cada uno. Nº triangulos= 2nm. En total 12(n+1)(m+1) +24nm.
- KB es 2^10
- MB es 2^20




