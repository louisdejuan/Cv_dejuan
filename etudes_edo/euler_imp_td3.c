#include <math.h>
#include "euler_imp.h"
#include <stdio.h>
#include <stdlib.h>


double* euler_imp(double coeff, double pas_de_temps, double cond_initiale,double cond_initiale_derivee, int nombre_de_pas)


{
	double* y_imp = NULL ;

	double a;
	a=1./(1.+pow(coeff*pas_de_temps,2));

	printf("je passe euler_imp \n ");

	printf("cond ini %lf , cond der %lf \n",cond_initiale,cond_initiale_derivee);

	

	



	y_imp = malloc ((nombre_de_pas+1) * sizeof(double)); // reservation de l’espace memoire
		if (y_imp == NULL)
			{
			printf("Impossible d’allouer le tableau y_imp \n");
			exit(0);
			}

	y_imp[0]=cond_initiale;
	y_imp[1]=cond_initiale_derivee;



	

for (int i=2; i<nombre_de_pas ; i++)

	{

		y_imp[i]=2.*a*y_imp[i-1]-a*y_imp[i-2];
		printf("ca %lf \n" ,y_imp[i]);

	}

	


	return y_imp;



}