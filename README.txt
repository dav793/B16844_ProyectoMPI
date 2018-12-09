
UNIVERSIDAD DE COSTA RICA
ESCUELA DE CIENCIAS DE LA COMPUTACION E INFORMATICA
PROYECTO MPI

AUTOR
David Vargas Barrantes - B16844


#   COMO CORRER:

    1. Compile
        mpic++ -std=c++11 -o B16844 ./B16844.cpp

    2. Corra con parámetros para exp1
        mpiexec -np <CANTIDAD DE PROCESOS> -f maq_mpi ./B16844 1000000 0.65 20 0.50 0 100000 500 1000

    3. Corra con parámetros para exp2
        mpiexec -np <CANTIDAD DE PROCESOS> -f maq_mpi ./B16844 10000000 0.65 20 0.50 0 1000000 1000 1000


#   PARAMETROS DE ENTRADA:

    1. Cantidad de personas
    2. Probabilidad de infección
    3. Duración de infección
    4. Probabilidad de recuperación
    5. Probabilidad de muerte - ELIMINADO, DEJE EN 0 -
    6. Cantidad de infectados inicial
    7. Tamaño del mundo
    8. Cantidad máxima de ticks a ejecutar


#   SALIDA

    Ver ./out.txt


