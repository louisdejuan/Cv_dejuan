#include <math.h>
#include "euler_imp.h"
#include <stdio.h>
#include <stdlib.h>
#include "newton.h"
#include "f1.h"


double* euler_imp(int nombre_de_pas,double pas_de_temps, double alpha, double cond_initiale)


{
	double* y_imp = NULL ;
	y_imp = malloc ((nombre_de_pas+1) * sizeof(double)); // reservation de l’espace memoire
		if (y_imp == NULL)
			{
			printf("Impossible d’allouer le tableau y_imp \n");
			exit(0);
			}

	y_imp[0]=cond_initiale;
	



	double copie;
	double A;
	double B;

for (int i=0; i<nombre_de_pas ; i++)

	{

		A=(pas_de_temps*alpha)/((cond_initiale+(i+1)*pas_de_temps)*exp(-alpha*(i+1)*pas_de_temps));

		B=-y_imp[i]-pas_de_temps*exp(-(i+1)*alpha*pas_de_temps);

		

		copie=newton(100,0.000000001,y_imp[i],A,B,f1,deriv_f1);

		y_imp[i+1]=copie;

		
	}

	


	return(y_imp);



}