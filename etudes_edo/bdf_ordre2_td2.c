#include <math.h>
#include "bdf_ordre2.h"
#include <stdio.h>
#include <stdlib.h>


double bdf_ordre2(double coeff, double pas_de_temps, double cond_initiale, int nombre_de_pas, double y_next_1)
{
	double y_next;
	
	double y_copie;

	



y_next=cond_initiale;

for (int i=0; i<nombre_de_pas-1 ; i++)

	{
		y_copie=y_next_1;
		y_next_1=(1/(3-2*coeff*pas_de_temps))*(-y_next+4*y_next_1);
		y_next=y_copie;



	}


	return y_next_1;







}
