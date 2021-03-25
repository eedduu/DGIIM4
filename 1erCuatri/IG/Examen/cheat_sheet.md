# Cheat sheet Informática Gráfica

## Visualizacion de primitivas

###Relleno/lineas/puntos

glPolygonMode(GL_FRON_AND_BACK, modo);

_modo_ es de tipo GLenum
GL_POINT->puntos
GL_LINE->lineas
GL_FILL-> relleno

Por defecto es GL_FILL

### Tipos primitiva

GL_TRIANGLES->triangulos 
GL_POINTS->puntos 
GL_LINES->lineas (segmentos)
GL_LINE_STRIP-> polilinea abierta
GL_LINE_LOOP->polilínea cerrada

GL_TRIANGLE_STRIP-> tira de triangulos (formando una especie de rectangulo)
GL_TRIANGLE_FAN-> abanico de triangulos
GL_QUAD_STRIP-> Tira de cuadrilateros
GL_POLYGON->polígono cualquiera
GL_QUADS->cuadrados

Estos tres últimos son obsoletos y no existen desde OpenGL 3.0

### Uso del modo inmediato con Begin y End

```c++
glBegin(tipo_primitiva) // tipo_primitiva puede ser GL_TRIANGLES, GL_POINTS, GL_LINES, GL_LINE_STRIP, GL_LINE_LOOP
    // Escribir sucesión de atributos del vértice + vértice. Ejemplo:
    glColor3f(r1, g1, b1); glNormal(xn1, yn1, zn1); glTexCoord2f(u1, v1); glVertex3f(x1, y1, z1);
    // glColor3f(), glNormal(), glTexCoord2f() pueden ser omitidos.
        ...
    glColor3f(rn, gn, bn); glNormal(xnn, ynn, znn); glTexcoord2f(un, vn); glVertex3f(xn, yn, zn);
glEnd()
```

### Uso del modo inmediato con Drawarrays / Drawelements

```c++
glEnableClientState(GL_VERTEX_ARRAY);

glBindVertexArray(0);
glVertexPointer(num_val_tupla, GL_FLOAT, 0, vertices.data());

// Si no hay indices
glDrawArrays(tipo_prim, 0, vertices.size());
//Si hay indices
glDrawElements(tipo_prim, indices.size(), GL_INT, ptr);
// tipo_prim puede valer GL_POLYGON, GL_QUADS, GL_TRIANGLES, GL_POINTS, GL_LINES, GL_LINE_STRIP, GL_LINE_LOOP, GL_TRIANGLE_STRIP, GL_QUAD_STRIP, GL_TRIANGLE_FAN
//ptr apunta al primer byte de los datos de indices. (indices->datos())

glDisableClientState(GL_VERTEX_ARRAY);
```

### Generación y uso de VAO y VBO

```c++
// 1\. Generate VBO if it does not exist (stores in VRAM)
GLuint VBO;
glGenBuffers(1, &VBO);

// 2\. Generate VAO if it does not exist (in order to refer the VBO)
GLuint VAO;
glGenVertexArrays(1, &VAO);

// 3\. Generate index table if needed
unsigned int EBO;
glGenBuffers(1, &EBO);

// 4\. bind Vertex Array Object
glBindVertexArray(VAO);

// 5\. copy our vertices array in a buffer for OpenGL to use
glBindBuffer(GL_ARRAY_BUFFER, VBO);
glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

// 6\. copy our index array (if it is not empty) in a element buffer for OpenGL to use
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

// 7\. then set our vertex attributes pointers
glVertexPointer(3, GL_FLOAT, 0, ptr_offset = 0);
    // Alternatively, use glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

glEnableClientState(GL_VERTEX_ARRAY) //glEnableVertexAttribArray(0);

// Keep in mind you don't need to generate the VAO or the VBO if you already had them. You should always try to reuse them.
// [...]

// ..:: Drawing code (in render loop) :: ..
// 8\. draw the object
glBindVertexArray(VAO);
someOpenGLFunctionThatDrawsOurTriangle();
    // e.g. : glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0)
glBindVertexArray(0);
```

##Arrayverts y descriptores
Arrayverts es una clase que encapsula una secuencia de vertices junto con sus atributos. Las coordenadas de los vertices y sus atributos están representadas mediante Descriptores de tabla. Pag 95 tema 1.
Tres metodos de visualizacion:
visualizarGL_MI_BVE(tipo_prim); // inmediato begin end
visualizarGL_MI_DAE(tipoprim); // inmediato con drawarray
visualizarGL_MD_VAO(tipoprim); // modo diferido

Clase Descriptores_tabla en transparencias 77-79 tema 1
Clase Arrayvertices en transparencias 84-85 tema 1


## Habilitar deshabilitar tablas
Una tabla es basicamente una secuencia de vertices (tabla vertices) o de atributos de vertices (colores, normales, textura). Si estoy usando descriptores de tabla, tengo los metodos activar_mi() y activar_md() para cada descriptor de tabla, y el metodo deshabilitar_tablas() de arrayverts que deshabilita todas.


### Habilitar
glBindVertexArray(0);

glVertexPointer(vum_vals_tupla, tipo_valores,0, puntero);
glTexCoordPointer(vum_vals_tupla, tipo_valores,0, puntero);
glColorPointer(vum_vals_tupla, tipo_valores,0, puntero);
glNormalPointer(tipo_valores,0 ,puntero);

glEnableClientState(ATRIBUTO); //atributo es GL_VERTEX_ARRAY, GL_COLOR_ARRAY, GL_TEXTURE_COORD_ARRAY, GL_NORMAL_ARRAY

Puntero apunta al primer byte de los elementos de la tabla. 

Si es en modo diferido, mirar pagina 103 tema 1 teoria.

###Deshabilitar
Simplemente hago: glDisableClientState(ATRIBUTO);


## Llamadas clave y parámetros

- `glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)`

  - GL_STREAM_DRAW: the data is set only once and used by the GPU at most a few times.
  - GL_STATIC_DRAW: the data is set only once and used many times.
  - GL_DYNAMIC_DRAW: the data is changed a lot and used many times.

- `void glDrawElements(GLenum mode, GLsizei count, GLenum type, const GLvoid * indices)`

  - mode Especifica qué primitivas se van a dibujar: GL_POINTS, GL_LINE_STRIP GL_LINE_LOOP, GL_LINES, GL_LINE_STRIP_ADJACENCY, GL_LINES_ADJACENCY, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN, GL_TRIANGLES, GL_TRIANGLE_STRIP_ADJACENCY, o GL_PATCHES
  - `count` Número de elementos que se van a dibujar
  - `type` Tipo de los indices ( GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, o GL_UNSIGNED_INT)
  - `indices` Puntero al vector donde están los índices almacenados.

- `void glDrawArrays(GLenum mode, GLint first, GLsizei count)`

  - `mode`. Especifica cómo interpreter la secuencia de datos ( GL_POINTS, GL_LINE_STRIP,GL_LINE_LOOP, GL_LINES, GL_LINE_STRIP_ADJACENCY, GL_LINES_ADJACENCY, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN, GL_TRIANGLES, GL_TRIANGLE_STRIP_ADJACENCY, GL_TRIANGLES_ADJACENCY o GL_PATCHES)
  - `first` Especifica el índice inicial de los array habilitados
  - `count` Especifica el número de elementos del array que se van a usar

- `void glVertexPointer (GLint size, GLenum type, GLsizei stride, const GLvoid * pointer)`

  - `size` Número de coordenadas por vértice: 2, 3 o 4.
  - `type` Tipo de cada coordenada del array (GL_SHORT, GL_INT, GL_FLOAT, o GL_DOUBLE)
  - `stride` Offset en bytes entre dos vertices consecutivos. Si es 0, se entiende que los vértices están consecutivos.
  - `pointer` Puntero a la primera coordenada del primer vértice del array

## Otras utilidades:

```c++
glLineWidth();
glPointSize();
glPolygonMode(cara, modo)
    // Cara: GL_FRONT, GL_BACK, GL_FRONT_AND_BACK.
    // Modo: GL_POINT, GL_LINE, GL_FILL
```

## Carga de texturas

```c++
void cargartextura(imagen.jpog){
    imagen = LeerArchivoJPEG(imagen.c_str() , ancho, alto);
    GLUint idtex;
    glGenTextures(1, &idtex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, idtex);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR ); //Parametros de textura
   //Envio a GPU
    gluBuild2DMipmaps( GL_TEXTURE_2D, GL_RGB, ancho, alto, GL_RGB, GL_UNSIGNED_BYTE, imagen);
    // textura 2d, formato interno, núm. de columnas (arbitrario) (GLsizei), núm de filas (arbitrario) (GLsizei), formato y orden de los texels en RAM, tipo de cada componente de cada texel,puntero a los bytes con texels
}
```

## Ejemplo de iluminación

```c++
void esfera_iluminada_colorverde_brillos_amarillos{
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    const float caf[4] = {r,g,b,1.0}, // color ambiental de la fuente
                cdf[4] = {r,g,b,1.0}, // color difuso de la fuente
                csf[4] = {r,g,b,1.0}; // color pseudoespecular de la fuente

    caf = verde;
    csf = amarillo;

    glLightfv(GL_LIGHT0, GL_AMBIENT, caf);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, cdf);  // No hace falta; no hay color fijo
    glLightfv(GL_LIGHT0, GL_SPECULAR, csf);

    const GLfloat dirf[4] = { 1.0, 1.0, 1.0 , 0.0 } ; // dirección de la luz posicional (0 => dirección, 1 => posición)
    glLightfv( GL_LIGHT0, GL_POSITION, dirf ); // hacemos la luz posicional con direccion

    glLightModeli( GL_LIGHT_MODEL_LOCAL_VIEWER, GL_FALSE ); // GL_FALSE => posicionada en el infinito, GL_TRUE => local
    //
    glShadeModel(GL_SMOOTH);
}
```

##Matriz de vista
Para fijar matriz de vista:
const GLfloat V[4][4]= 4 arrays de coordenadas (x,y,z, origen)

glMatrixMode(GL_MODELVIEW);
glLoadIdentity();
glMultMatrix(V);

###Calculo matriz vista

##Matriz de proyeccion
Matriz de pryeccion P 4x4

###Cauce fijo
glMatrixMode(GL_PROJECTION);
glLoadIdentity();
glMultMatrixf(matrizProyec);

###Cauce programable
glUseProgram(id_prog);
glUniformMatrix4fv(localizacion_matproyec, 1, GL_FALSE, mat_proyeccion);


