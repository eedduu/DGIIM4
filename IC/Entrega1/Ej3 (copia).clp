(deffacts Ramas
  (Rama Computacion_y_Sistemas_Inteligentes)
  (Rama Ingenieria_del_Software)
  (Rama Ingenieria_de_Computadores)
  (Rama Sistemas_de_Informacion)
  (Rama Tecnologias_de_la_Informacion)
)


(deffacts expertos
  (Experto Edu)
)

(deftemplate Consejo
  (field Rama)
  (multifield Explicacion)
  (multifield Experto)
)

(defrule Empieza
  =>
  (printout t "Empecemos, te gustan las matematicas? (Si/No):" crlf)
  (assert (Mates (read)))
)

(defrule Mates
  (Mates Si)
  =>
  (printout t "Y te gusta programar ? (Si/No)" crlf)
  (assert (Prog (read)))
)

(defrule Hardware
  (Mates No)
  (or (nota Media) (nota Alta))
  =>
  (printout t "Te gusta el Hardware? " )
  (assert (Hardware (read)))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defrule MatesNs
  (Mates "No se")
  =>
  (printout t "Y te gusta programar? (Si/No)" crlf)
  (assert (Prog (read)))
)

(defrule NotaNs
  (Mates "No se")
  (Prog ?c)
  =>
  (printout t "¿Cual es tu nota media (1-10)?" )
  (assert (ponerNota(read)))
)

(defrule HardwareNs
  (Mates "No se")
  (Prog ?c)
  (nota ?c)
  =>
  (printout t "Te gusta el Hardware? " )
  (assert (Hardware (read)))

)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defrule NotaMedia
  (Mates No)
  =>
  (printout t "¿Cual es tu nota media (1-10)?" )
  (assert (ponerNota(read)))
)

;;Notas
(defrule notaBaja
  (ponerNota ?c)
  (test (<= ?c 6))
  =>
  (assert (nota Baja))
)

(defrule notaMedia
  (ponerNota ?c)
  (test (> ?c 6))
  (test (<= ?c 8.5))
  =>
  (assert (nota Media))
)

(defrule notaAlta
  (ponerNota ?c)
  (test (> ?c 8.5))
  =>
  (assert (nota Alta))
)


(defrule eligeCSI
  (or (Mates Si) (Mates "No se"))
  (Prog Si)
  (nota Alta)

  =>
  (assert (elegido Computacion_y_Sistemas_Inteligentes))
)

(defrule eligeTI
  (Mates Si)
  (Prog No)
  =>
  (assert (elegido Tecnologias_de_la_Informacion))
)

(defrule eligeIS
  (Mates No)
  (nota Baja)
  =>
  (assert (elegido Ingenieria_del_Software))
)

(defrule eligeIC
  (Mates No)
  (or (nota Media) (nota Alta))
  (Hardware Si)
  =>
  (assert (elegido Ingenieria_de_Computadores))
)

(defrule eligeSI
  (Mates No)
  (or (nota Media) (nota Alta))
  (Hardware No)
  =>
  (assert (elegido Sistemas_de_Informacion))

)


(defrule Explicaciones1
  (elegido ?c)
  (Mates Si)
  (Prog Si)
  =>
  (assert (Explicacion "te gustan las mates y la programacion"))
)

(defrule Explicaciones2
  (elegido ?c)
  (Mates Si)
  (Prog No)
  =>
  (assert (Explicacion "te gustan las mates pero no la programacion"))
)

(defrule Explicaciones3
  (elegido ?c)
  (Mates No)
  (nota Baja)
  =>
  (assert (Explicacion "no te gustan las mates y tienes la nota media baja"))
)

(defrule Explicaciones4
  (elegido ?c)
  (Mates No)
  (or (nota Media) (nota Alta))
  (Hardware Si)
  =>
  (assert (Explicacion "no te gustan las mates, no tienes mala nota y ademas te gusta el Hardware"))
)

(defrule Explicaciones5
  (elegido ?c)
  (Mates No)
  (or (nota Media) (nota Alta))
  (Hardware No)
  =>
  (assert (Explicacion "no te gustan las mates, no tienes mala nota y ademas no te gusta el hardware"))
)


(defrule consejito
  (elegido ?c)
  (Explicacion ?y)
  =>
  (assert (Consejo (Rama ?c) (Explicacion ?y) (Experto Edu)))
)

(defrule imprimir
  (Consejo (Rama ?c) (Explicacion ?y) (Experto ?x))
  =>
  (printout t "Hola, soy el experto " ?x crlf)
  (printout t "Te recomiendo la rama " ?c crlf)
  (printout t "Te la recomiendo porque " ?y crlf)
)
