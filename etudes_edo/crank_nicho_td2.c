#include <math.h>
#include "crank_nicho.h"
#include <stdio.h>
#include <stdlib.h>

double crank_nicho(double coeff, double pas_de_temps, double cond_initiale, int nombre_de_pas)
{
	double y_next;

	



y_next=cond_initiale;
for (int i=0; i<nombre_de_pas ; i++)

	{

		y_next=y_next*(1+(pas_de_temps*coeff/2))/(1-(pas_de_temps*coeff/2));



	}


	return y_next;







}
