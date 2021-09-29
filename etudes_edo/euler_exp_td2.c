#include <math.h>
#include "euler_exp.h"
#include <stdio.h>
#include <stdlib.h>


double euler_exp(double coeff, double pas_de_temps, double cond_initiale, int nombre_de_pas)
{
	double y_next;

	

	if((coeff<0)&&(pas_de_temps>(-1/coeff)))
	{
		printf("erreur");
		exit (0);
	}

y_next=cond_initiale;
for (int i=0; i<nombre_de_pas ; i++)

	{

		y_next+=coeff*y_next*pas_de_temps;



	}


	return y_next;



}