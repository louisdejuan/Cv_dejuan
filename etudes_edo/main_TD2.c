#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "euler_imp.h"
#include "euler_exp.h"
#include "crank_nicho.h"
#include "bdf_ordre2.h"
int main(int argc, char *argv[])

	{
		if (argc != 2) {printf("\nLe programme doit recevoir 2 arguments\n");return(EXIT_FAILURE);}


		int nombre_de_pas;
		double temps_final ;
		double coeff ;
		double cond_initiale ;

		

		FILE* fichier=fopen(argv[1],"r");

			if (fichier == NULL)
			{
				printf("Impossible d’ouvrir le fichier %s \n",argv[1]);return(EXIT_FAILURE);
			}

		fscanf(fichier, "%d %lf %lf %lf", &nombre_de_pas , &temps_final, &coeff , &cond_initiale);	

		
		double pas_de_temps;double y_next;
		double y_reel;
		double y_next_1;


		pas_de_temps=temps_final/nombre_de_pas;
		y_reel=cond_initiale*exp(temps_final*coeff);


		y_next=euler_exp(coeff,pas_de_temps,cond_initiale,nombre_de_pas);
		printf("le reslutat est eulerexp %f, la reelle est %f \n",y_next,y_reel);

		y_next=euler_imp(coeff,pas_de_temps,cond_initiale,nombre_de_pas);
		printf("le reslutat est eulerimp  %f, la reelle est %f \n",y_next,y_reel);

		y_next=crank_nicho(coeff,pas_de_temps,cond_initiale,nombre_de_pas);
		printf("le reslutat est  crank_nicho  %f, la reelle est %f \n",y_next,y_reel);

		y_next_1=crank_nicho(coeff,pas_de_temps,cond_initiale,1);

		y_next=bdf_ordre2(coeff,pas_de_temps,cond_initiale,nombre_de_pas,y_next_1);
		printf("le reslutat est  bdf_ordre2  %f, la reelle est %f \n",y_next,y_reel);




		return 0;

	}