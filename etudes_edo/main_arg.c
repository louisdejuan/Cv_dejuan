#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "euler_imp.h"
#include "euler_exp.h"
#include "crank_nicho.h"

int main(int argc, char *argv[])

	{
		if (argc != 5) {printf("\nLe programme doit recevoir quatre arguments\n");return(EXIT_FAILURE);}


		int nombre_de_pas = atoi(argv[1]);
		double temps_final = atof(argv[2]);
		double coeff = atof(argv[3]);
		double cond_initiale = atof(argv[4]);

		
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