#include <math.h>
#include "euler_exp.h"
#include <stdio.h>
#include <stdlib.h>


double* euler_exp(double coeff, double pas_de_temps, double cond_initiale,double cond_initiale_derivee, int nombre_de_pas)
{
	double* y_exp = NULL ;


	y_exp = malloc ((nombre_de_pas+1) * sizeof(double)); // reservation de l’espace memoire
		if (y_exp == NULL)
			{
			printf("Impossible d’allouer le tableau y_exp \n");
			exit(0);
			}

	y_exp[0]=cond_initiale;
	y_exp[1]=y_exp[0]+pas_de_temps*cond_initiale_derivee;


	

for (int i=2; i<nombre_de_pas ; i++)

	{

		y_exp[i]=-pow(pas_de_temps,2)*y_exp[i-2]*pow(coeff,2)+2*y_exp[i-1]-y_exp[i-2];

	}


	return y_exp;



}