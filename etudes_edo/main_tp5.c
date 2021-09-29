#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "euler_imp.h"
#include "euler_exp.h"
#include "crank_nicho.h"

int main(int argc, char *argv[])

	{
		if (argc != 3) {printf("\nLe programme doit recevoir 3 arguments\n");return(EXIT_FAILURE);}


		int nombre_de_pas;
		double temps_final ;
		double coeff ;
		double condition_der ;
	
		

		

		FILE* fichier_lec=fopen(argv[1],"r");

			if (fichier_lec == NULL)
			{
				printf("Impossible d’ouvrir le fichier %s \n",argv[1]);return(EXIT_FAILURE);

			}

		fscanf(fichier_lec, "%d %lf %lf %lf", &nombre_de_pas , &temps_final, &coeff , &condition_der );	

		FILE* fichier_ecr = fopen(argv[2],"w+");

		double pas_de_temps;
		pas_de_temps=temps_final/nombre_de_pas;




		double* y_exp = NULL; // y_imp est un pointeur vers un double
		

		double* y_imp = NULL; // y_imp est un pointeur vers un double


		double* y_cni = NULL; // y_imp est un pointeur vers un double

		double condition_ini =0;
		
		

		y_exp=euler_exp(nombre_de_pas,pas_de_temps,coeff,condition_ini,condition_der);
			

		y_imp=euler_imp(nombre_de_pas,pas_de_temps,coeff,condition_ini,condition_der);
		

		y_cni=crank_nicho(nombre_de_pas,pas_de_temps,coeff,condition_ini,condition_der);



		


	



		
		double temps_ecoule;
		
		temps_ecoule=0;


		for (int i=0;i<nombre_de_pas;i++)
		{
			
			fprintf(fichier_ecr, "%g %g %g %g \n", temps_ecoule, y_exp[i], y_imp[i], y_cni[i] );
			temps_ecoule=temps_ecoule+pas_de_temps;

		}   
		

		free (y_imp);
		free (y_exp);
		free (y_cni);
	


		fclose(fichier_lec);
		fclose(fichier_ecr);


		return 0;

	}