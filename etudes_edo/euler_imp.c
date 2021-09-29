#include <math.h>
#include "euler_imp.h"
#include <stdio.h>
#include <stdlib.h>
#include "newton.h"
#include "f1.h"


double* euler_imp(int nombre_de_pas,double pas_de_temps, double coeff, double cond_initiale,)


{
	double* y_imp = NULL ;

	double b;
	b=-coeff*pow(pas_de_temps,2);



	

	y_imp = malloc ((nombre_de_pas+1) * sizeof(double)); // reservation de l’espace memoire
		if (y_imp == NULL)
			{
			printf("Impossible d’allouer le tableau y_imp \n");
			exit(0);
			}

	y_imp[0]=cond_initiale;
	



	double copie;

for (int i=2; i<nombre_de_pas ; i++)

	{

		copie=newton(y_imp[i-1],y_imp[i-1],y_imp[i-2],b,0.0000001,100,1);

		y_imp[i]=copi;

		
		
	}

	


	return(y_imp);



}