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
		int k;

		

		FILE* fichier_lec=fopen(argv[1],"r");

			if (fichier_lec == NULL)
			{
				printf("Impossible d’ouvrir le fichier %s \n",argv[1]);return(EXIT_FAILURE);
			}

		fscanf(fichier_lec, "%d %lf %lf %lf %d", &nombre_de_pas , &temps_final, &coeff , &cond_initiale, &k);	

		FILE* fichier_ecr = fopen(argv[2],"w+");

		double pas_de_temps;
		double y_next;
		double y_reel;
		double y_next_1;
		double e_bdf;double e_exp ;double e_cni; double e_imp;
		double nombre_de_pas_itere;



		y_reel=cond_initiale*exp(temps_final*coeff);


		
		
		

		
		for (int i=k;i>0;i--)
		{
			nombre_de_pas_itere=nombre_de_pas*pow(2,i);
			pas_de_temps=temps_final/nombre_de_pas_itere;

			printf("pas_de_temps = %g , nombre_de_pas= %g \n", pas_de_temps, nombre_de_pas_itere );


			y_next=euler_exp(coeff,pas_de_temps,cond_initiale,nombre_de_pas_itere);
			e_exp=fabs(y_next-y_reel);

			y_next=euler_imp(coeff,pas_de_temps,cond_initiale,nombre_de_pas_itere);
			e_imp=fabs(y_next-y_reel);

			y_next=crank_nicho(coeff,pas_de_temps,cond_initiale,nombre_de_pas_itere);
			e_cni=fabs(y_next-y_reel);

			y_next_1=crank_nicho(coeff,pas_de_temps,cond_initiale,1);
			y_next=bdf_ordre2(coeff,pas_de_temps,cond_initiale,nombre_de_pas_itere,y_next_1);
			e_bdf=fabs(y_next-y_reel);

			fprintf(fichier_ecr, "%g %g %g %g %g \n", pas_de_temps, e_exp, e_imp, e_cni, e_bdf);




		}
		


		
		

		fclose(fichier_lec);
		fclose(fichier_ecr);


		return 0;

	}