#include <math.h>
#include "crank_nicho.h"
#include <stdio.h>
#include <stdlib.h>


double* crank_nicho(double coeff, double pas_de_temps, double cond_initiale,double cond_initiale_derivee, int nombre_de_pas)
{
	double* y_cni=NULL ;
	double alpha=(1/4.)*pow(coeff*pas_de_temps,2);

	y_cni = malloc ((nombre_de_pas+1) * sizeof(double)); // reservation de l’espace memoire
		if (y_cni == NULL)
			{
			printf("Impossible d’allouer le tableau y_cni \n");
			exit(0);
			}

	y_cni[0]=cond_initiale;
	y_cni[1]=y_cni[0]+pas_de_temps*cond_initiale_derivee;


	

for (int i=2; i<nombre_de_pas ; i++)

	{

		y_cni[i]=(2.*(1.-alpha)*y_cni[i-1]-(1.+alpha)*y_cni[i-2])/(1.+alpha);

	}


	return y_cni;



}