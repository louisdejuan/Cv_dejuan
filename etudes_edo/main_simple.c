#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "euler_imp.h"
#include "euler_exp.h"
#include "crank_nicho.h"
#include "bdf_ordre2.h"


int main()
	{
		double coeff=-1; double temps_final=10; double cond_initiale=10; int nombre_de_pas=1000;
		double pas_de_temps;double y_next;
		double y_reel;


		pas_de_temps=temps_final/nombre_de_pas;
		y_reel=cond_initiale*exp(temps_final*coeff);


		y_next=euler_exp(coeff,pas_de_temps,cond_initiale,nombre_de_pas);
		printf("le reslutat est eulerexp %f, la reelle est %f \n",y_next,y_reel);

		y_next=euler_imp(coeff,pas_de_temps,cond_initiale,nombre_de_pas);
		printf("le reslutat est eulerimp  %f, la reelle est %f \n",y_next,y_reel);

		y_next=crank_nicho(coeff,pas_de_temps,cond_initiale,nombre_de_pas);
		printf("le reslutat est eulerimp  %f, la reelle est %f \n",y_next,y_reel);

		return 0;

	}