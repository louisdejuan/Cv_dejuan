#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "euler_imp.h"
#include "euler_exp.h"
#include "crank_nicho.h"
#include "bdf_ordre2.h"
int main(int argc, char *argv[])

	{
		if (argc != 3) {printf("\nLe programme doit recevoir 2 arguments\n");return(EXIT_FAILURE);}


		int nombre_de_pas;
		double temps_final ;
		double coeff ;
		double cond_initiale ;
		double cond_initiale_derivee ;
		

		

		FILE* fichier_lec=fopen(argv[1],"r");

			if (fichier_lec == NULL)
			{
				printf("Impossible d’ouvrir le fichier %s \n",argv[1]);return(EXIT_FAILURE);
			}

		fscanf(fichier_lec, "%d %lf %lf %lf %lf", &nombre_de_pas , &temps_final, &coeff , &cond_initiale,&cond_initiale_derivee);	

		FILE* fichier_ecr = fopen(argv[2],"w+");

		double pas_de_temps;
		pas_de_temps=temps_final/nombre_de_pas;




		double* y_exp = NULL; // y_imp est un pointeur vers un double
		

		double* y_imp = NULL; // y_imp est un pointeur vers un double


		double* y_cni = NULL; // y_imp est un pointeur vers un double

		double* y_reel = NULL;
		

		y_exp=euler_exp(coeff,pas_de_temps,cond_initiale,cond_initiale_derivee,nombre_de_pas);
			

		y_imp=euler_imp(coeff,pas_de_temps,cond_initiale,cond_initiale_derivee,nombre_de_pas);
		

		y_cni=crank_nicho(coeff,pas_de_temps,cond_initiale,cond_initiale_derivee,nombre_de_pas);



		y_reel = malloc ((nombre_de_pas+1) * sizeof(double)); // reservation de l’espace memoire

		if (y_reel == NULL)
			{
			printf("Impossible d’allouer le tableau y_reel \n");
			return(EXIT_FAILURE);
			}



	



		
		double temps_ecoule;
		pas_de_temps=temps_final/nombre_de_pas;
		temps_ecoule=0;


		for (int i=0;i<nombre_de_pas;i++)
		{
			y_reel[i]=cond_initiale*cos(coeff*temps_ecoule)+(cond_initiale_derivee/coeff)*sin(coeff*temps_ecoule);
			fprintf(fichier_ecr, "%g %g %g %g %g \n", temps_ecoule, y_reel[i], y_exp[i], y_imp[i], y_cni[i] );
			temps_ecoule=temps_ecoule+pas_de_temps;

		}   
		
		free (y_imp);
		free (y_exp);
		free (y_cni);
		free (y_reel);




		


		
		

		fclose(fichier_lec);
		fclose(fichier_ecr);


		return 0;

	}