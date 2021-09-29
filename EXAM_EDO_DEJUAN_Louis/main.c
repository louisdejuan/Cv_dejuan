#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "euler_imp.h"
#include "euler_exp.h"

int main(int argc, char *argv[])

	{
		if (argc != 3) {printf("\nLe programme doit recevoir 3 arguments\n");return(EXIT_FAILURE);}


		int nombre_de_pas;

		double temps_final ;

		double alpha ;

		double condition_ini ;
		
		
	
		

		

		FILE* fichier_lec=fopen(argv[1],"r");

			if (fichier_lec == NULL)
			{
				printf("Impossible d’ouvrir le fichier %s \n",argv[1]);return(EXIT_FAILURE);

			}

		fscanf(fichier_lec, "%d %lf %lf %lf", &nombre_de_pas , &temps_final, &alpha , &condition_ini);	

		FILE* fichier_ecr = fopen(argv[2],"w+");

		double pas_de_temps;
		pas_de_temps=temps_final/nombre_de_pas;




		double* y_exp = NULL; // y_imp est un pointeur vers un double
		

		double* y_imp = NULL; // y_imp est un pointeur vers un double


		double* y_reel = NULL; 

		y_reel = malloc ((nombre_de_pas+1) * sizeof(double)); // reservation de l’espace memoire
		if (y_reel == NULL)
		{
			printf("Impossible d’allouer le tableau y_reel \n");
			exit(0);
		}
		
		

		y_exp=euler_exp(nombre_de_pas,pas_de_temps,alpha,condition_ini);
			

		y_imp=euler_imp(nombre_de_pas,pas_de_temps,alpha,condition_ini);
		

		



		


	



		
		double temps_ecoule;
		
		temps_ecoule=0;

		double erreur_imp=0;
		double erreur_exp=0;



		for (int i=0;i<nombre_de_pas;i++)
		{
			y_reel[i]=(condition_ini+temps_ecoule)*exp(-alpha*temps_ecoule);

			erreur_imp=erreur_imp+pow(y_imp[i]-y_reel[i],2);
			erreur_exp=erreur_exp+pow(y_exp[i]-y_reel[i],2);

			

			fprintf(fichier_ecr, "%g %g %g %g \n", temps_ecoule, y_imp[i], y_exp[i],y_reel[i]);

			temps_ecoule=temps_ecoule+pas_de_temps;

		}   

		erreur_exp=pow(pas_de_temps*erreur_exp,0.5);
		erreur_imp=pow(pas_de_temps*erreur_imp,0.5);

		
		printf("erreur_exp : %lf \n",erreur_exp);
		printf("erreur_imp : %lf \n",erreur_imp);

		free (y_imp);
		
		free (y_exp);
		
		free (y_reel);
		
	


		fclose(fichier_lec);
		fclose(fichier_ecr);


		return 0;

	}