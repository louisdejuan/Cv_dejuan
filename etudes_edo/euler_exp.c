#include <math.h>
#include "euler_exp.h"
#include <stdio.h>
#include <stdlib.h>
#include "f1.h"



double* euler_exp(int nombre_de_pas,double pas_de_temps, double alpha, double cond_initiale )

{

	

	y_exp = malloc ((nombre_de_pas+1) * sizeof(double)); // reservation de l’espace memoire
		if (y_exp == NULL)
			{
			printf("Impossible d’allouer le tableau y_exp \n");
			exit(0);
			}

	y_exp[0]=cond_initiale;


double y_aux;

	

for (int i=1; i<nombre_de_pas ; i++)

	{	

		y_aux=y_exp[i]+pas_de_temps*f(i*pas_de_temps,y_exp[i],alpha,cond_initiale);
		y_exp[i+1]=y_exp[i]+(pas_de_temps/2)*(f(i*pas_de_temps,y_exp[i],alpha,cond_initiale)+f((i+1)*pas_de_temps,y_aux,alpha,cond_initiale));


	}


	return(y_exp);



}