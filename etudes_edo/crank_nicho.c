#include <math.h>
#include "crank_nicho.h"
#include <stdio.h>
#include <stdlib.h>
#include "newton.h"


double* crank_nicho(int nombre_de_pas,double pas_de_temps, double coeff, double cond_initiale, double cond_initiale_derivee)
{
	double* y_cni=NULL ;
	double b;
	b=-coeff*pow(pas_de_temps,2);

	y_cni = malloc ((nombre_de_pas+1) * sizeof(double)); // reservation de l’espace memoire
		if (y_cni == NULL)
			{
			printf("Impossible d’allouer le tableau y_cni \n");
			exit(0);
			}

	y_cni[0]=cond_initiale;
	y_cni[1]=y_cni[0]+pas_de_temps*cond_initiale_derivee;


double copi;

for (int i=2; i<nombre_de_pas ; i++)

	{

		copi=newton(y_cni[i-1],y_cni[i-1],y_cni[i-2],b,0.0000001,100,2);
		y_cni[i]=copi;
		
		

	}


	return(y_cni);



}