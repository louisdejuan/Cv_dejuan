#ifndef NEWTON_H
#define NEWTON_H
#include <math.h>
#include <stdlib.h>
#include <stdio.h>


double newton(int, double, double , double, double ,double (*fonc) (double, double, double), double (*deriv_fonc) (double, double));




// double (*fonc) (double) est un pointeur de fonction retournant
// un double et ayant comme argument un double.
// Plus g ́en ́eralement type_a (*f) (type_b, type_c) est un pointeur
// de fonction retournant une valeur de type type_a et ayant comme argument
// les types type_b et type_c



#endif