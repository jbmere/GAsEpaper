#! /bin/bash
# Caso 01 : Coste obtenido 222  Como Fadi
time python  machauto_01.py -s 200 -n 200 -i C3_0.csv -m 0.2 -c 0.85 > c3_0.out
# real	0m18.329s
# user	0m12.084s
# sys	0m0.656s
#
# Caso 02 : Coste obtenido 167  Muy difernete a Fadi en solución
time python  machauto_01.py -s 200 -n 200 -i C3_1.csv -m 0.2 -c 0.85 > c3_1.out
# real	0m20.598s
# user	0m13.656s
# sys	0m0.872s
#
# Caso 03 : Coste obtenido  203  Similar a la solucion del caso 01
time python  machauto_01.py -s 200 -n 200 -i C3_2.csv -m 0.2 -c 0.85 > c3_2.out
# real	0m18.042s
# user	0m11.620s
# sys	0m0.540s
#
# Caso 04 : C4 Fadi. Coste obtenido 393  el de Fadi
time python  machauto_01.py -s 1500 -n 100 -i C4_0.csv -m 0.6 -c 0.85 > c4_0.out
# real	2m8.720s
# user	1m18.240s
# sys	0m3.580s
#
# Caso 05 : C5 Fadi. Coste obtenido 395   ligeramente más alto que el de Fadi
# La explicación es que no permitimos transiciones instantaneas sino que
# la duración es al penos un periodo y la solucion de Fadi si acepta trans 
# de duracion 0
time python  machauto_01.py -s 1500 -n 100 -i C5_0.csv -m 0.6 -c 0.85 > c5_0.out
# real	1m48.031s
# user	1m17.680s
# sys	0m2.240s
#
# Caso 06 : C6 Fadi. Coste obtenido 342  Fejor que el que tiene Fadi.
# Rotacion de Jobs
time python  machauto_01.py -s 1500 -n 100 -i C6_0.csv -m 0.6 -c 0.85 > c6_0.out
# real	2m0.371s
# user	1m21.024s
# sys	0m2.412s

