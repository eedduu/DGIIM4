;He realizado una modificación del código visto en clase para que, además de inferir relaciones entre familiares
; pueda responder a la pregunta de, dado un familiar, dar familiares suyos (abuelos, nietos, primos...)

; El programa pide el nombre de un familiar, y un tipo de relación: ABUELO, NIETO, ESPOSO, YERNO...

(deffacts personas
  (hombre Edu)
  (hombre Ale)
  (hombre Eduardo)
  (hombre Ruperto)
  (hombre Antonio)
  (hombre Ruper)
  (hombre Pedro)
  (hombre Pablo)

  (mujer Covi)
  (mujer Nieves)
  (mujer Merci)
  (mujer Raquel)
  (mujer Lola)
  (mujer Gloria)
  (mujer Julia)
   )

(deftemplate Relacion
  (slot tipo (type SYMBOL) (allowed-symbols HIJO PADRE ABUELO NIETO HERMANO ESPOSO PRIMO TIO SOBRINO  CUNIADO YERNO SUEGRO))
  (slot sujeto)
  (slot objeto))

(deffacts relaciones
   (Relacion (tipo HIJO) (sujeto Edu) (objeto Eduardo)) ; "Luis es HIJO de Antonio"
   (Relacion (tipo HIJO) (sujeto Ale) (objeto Eduardo))
   (Relacion (tipo HIJO) (sujeto Eduardo) (objeto Pedro))
   (Relacion (tipo HIJO) (sujeto Antonio) (objeto Pedro))
   (Relacion (tipo HIJO) (sujeto Lola) (objeto Ruper))
   (Relacion (tipo HIJO) (sujeto Julia) (objeto Ruper))
   (Relacion (tipo HIJO) (sujeto Ruper) (objeto Ruperto))
   (Relacion (tipo HIJO) (sujeto Covi) (objeto Ruperto))
   (Relacion (tipo HIJO) (sujeto Pablo) (objeto Antonio))

   (Relacion (tipo ESPOSO) (sujeto Antonio) (objeto Raquel)) ; "Antonio es ESPOSO de Laura"
   (Relacion (tipo ESPOSO) (sujeto Eduardo) (objeto Covi))
   (Relacion (tipo ESPOSO) (sujeto Ruper) (objeto Gloria))
   (Relacion (tipo ESPOSO) (sujeto Ruperto) (objeto Merci))
   (Relacion (tipo ESPOSO) (sujeto Antonio) (objeto Raquel))
   (Relacion (tipo ESPOSO) (sujeto Pedro) (objeto Nieves)))

;;;;;;; DUALES

(deffacts duales
(dual HIJO PADRE) (dual ABUELO NIETO) (dual HERMANO HERMANO)
(dual ESPOSO ESPOSO)
(dual PRIMO PRIMO) (dual TIO SOBRINO)
(dual CUNIADO CUNIADO)
(dual YERNO SUEGRO))

;;;;;; COMPOSICIONES

(deffacts compuestos
(comp HIJO HIJO NIETO) (comp PADRE PADRE ABUELO) (comp ESPOSO PADRE PADRE)(comp HERMANO PADRE TIO) (comp HERMANO ESPOSO CUNIADO) (comp ESPOSO HIJO YERNO) (comp ESPOSO HERMANO CUNIADO) (comp HIJO PADRE HERMANO) (comp ESPOSO CUNIADO CUNIADO) (comp ESPOSO TIO TIO)  (comp HIJO TIO PRIMO)  )


;;;;;; FEMENINO

(deffacts femenino
(femenino HIJO HIJA) (femenino PADRE MADRE) (femenino ABUELO ABUELA) (femenino NIETO NIETA) (femenino HERMANO HERMANA) (femenino ESPOSO ESPOSA) (femenino PRIMO PRIMA) (femenino TIO TIA) (femenino SOBRINO SOBRINA) (femenino CUNIADO CUNIADA) (femenino YERNO NUERA) (femenino SUEGRO SUEGRA))


;;;;; REGLAS DEL SISTEMA ;;;;;

;;;; La dualidad es simetrica: si r es dual de t, t es dual de r. Por eso solo metimos como hecho la dualidad en un sentidos, pues en el otro lo podiamos deducir con esta regla

(defrule autodualidad
      (dual ?r ?t)
=>
   (assert (dual ?t ?r)))


;;;; Si  x es R de y, entonces y es dualdeR de x

(defrule dualidad
   (Relacion (tipo ?r) (sujeto ?x) (objeto ?y))
   (dual ?r ?t)
=>
   (assert (Relacion (tipo ?t) (sujeto ?y) (objeto ?x))))


;;;; Si  y es R de x, y x es T de z entonces y es RoT de z
;;;; a�adimos que z e y sean distintos para evitar que uno resulte hermano de si mismo y cosas asi.

(defrule composicion
   (Relacion (tipo ?r) (sujeto ?y) (objeto ?x))
   (Relacion (tipo ?t) (sujeto ?x) (objeto ?z))
   (comp ?r ?t ?u)
   (test (neq ?y ?z))
=>
   (assert (Relacion (tipo ?u) (sujeto ?y) (objeto ?z))))

;;;;; Como puede deducir que tu hermano es tu cu�ado al ser el esposo de tu cu�ada, eliminamos los cu�ados que sean hermanos

(defrule limpiacuniados
    (Relacion (tipo HERMANO) (sujeto ?x) (objeto ?y))
    ?f <- (Relacion (tipo CUNIADO) (sujeto ?x) (objeto ?y))
=>
	(retract ?f) )

;;;;; SOLICITO EL NOMBRE DE LA PERSONA

(defrule pideNombre
	; sin antecedente para que se ejecute el primero
	=>
	(printout t "Escriba el nombre de un miembro de la familia: ")
	(assert (pideRelDe (read)))
)

; SOLICITO LA RELACIÓN QUE QUIERO BUSCAR

; Añado un hecho "NoHay", para que en caso de que no haya ningún familiar del tipo requerido se indique

(defrule pideRelacion
  (pideRelDe ?A)
	=>
	(printout t "Que relacion quieres imprimir respecto de (nombre de la relacion en mayuscula entero): " ?A " ")
	(assert (buscaparentesco (read)))
  (assert (NoHay ?A) )

)

; Busco relaciones que contengan a dicha persona como sujeto. Si estoy buscando padres de Edu, buscaré relaciones donde Edu sea el hijo

; Tengo dos "fases" para buscar cada tipo de familiar. En la primera simplemente confirmo que existen familiares del tipo que estoy buscando, para poder
; eliminar el hecho "NoHay", de manera que no se ejecute esa regla. En la segunda ya selecciono los familiares que cumplen la regla.

(defrule buscar
  (buscaparentesco ?R)
  (pideRelDe ?A)
  (Relacion (tipo ?R) (sujeto ?B) (objeto ?A))
  ?f <- (NoHay ?b)
  =>
  (assert (encontrado a))
  (retract ?f)

)

(defrule buscar2
  (buscaparentesco ?R)
  (pideRelDe ?A)
  (Relacion (tipo ?R) (sujeto ?B) (objeto ?A))
  (encontrado a)
  =>
  (assert (primerapersona ?B))
  (assert (segundapersona ?A))

)

; Escribe las relaciones según sean femenino o masculino

(defrule relacionmasculino
  (primerapersona ?x)
  (segundapersona ?y)
  (Relacion (tipo ?r) (sujeto ?y) (objeto ?x))
  (hombre ?y)
=>
  (printout t ?y " es " ?r " de " ?x crlf)

  )

(defrule relacionfemenino2

  (primerapersona ?x)
  (segundapersona ?y)
  (Relacion (tipo ?r) (sujeto ?y) (objeto ?x))
  (mujer ?y)
  (femenino ?r ?t)
=>
  (printout t ?y " es " ?t " de " ?x crlf)
  )


; Regla en caso de que no haya ningun familiar de los buscados
(defrule NoHay
  (NoHay ?y)
  (buscaparentesco ?B)
  =>
  (printout t ?y " no tiene ningun " ?B crlf)
)
