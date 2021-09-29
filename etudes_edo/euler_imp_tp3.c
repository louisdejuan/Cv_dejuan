#include <math.h>
#include "euler_imp.h"
#include <stdio.h>
#include <stdlib.h>


double* euler_imp(double coeff, double pas_de_temps, double cond_initiale,double cond_initiale_derivee, int nombre_de_pas)
{
	double* y_imp = NULL ;

	y_imp = malloc ((nombre_de_pas+1) * sizeof(double)); // reservation de l’espace memoire
		if (y_imp == NULL)
			{
			printf("Impossible d’allouer le tableau y_imp \n");
			exit(0);
			}

	y_imp[0]=cond_initiale;
	y_imp[1]=y_imp[0]+pas_de_temps*cond_initiale_derivee;


	

for (int i=2; i<nombre_de_pas ; i++)

	{

		y_imp[i]=(2*y_imp[i-1]-y_imp[i-2])/(1+pow(coeff*pas_de_temps,2));

	}


	return y_imp;



}