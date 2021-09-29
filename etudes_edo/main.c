// fichier main.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "newton.h"
#include "f1.h"

int main()

{
int N;
double eps;
double x0 ;
double xres1;


N = 100; // ou `a lire dans un fichier de donn ́ees
eps = 1.e-6; // c.f. TPs pr ́ec ́edents
x0=1.0;


printf("Nombre maximal d iterations: %d \n",N);

printf("Tolerance: %f \n",eps);

printf("Itere initial: %f \n",x0);


xres1 = newton(N,eps,x0,f1,deriv_f1); 


// le nom d’une fonction sert

// aussi de pointeur vers cette fonction


printf("racine carree de deux %lf \n",xres1);

return 0;

}

