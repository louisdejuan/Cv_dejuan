#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tir_euler_imp.h"

int main(int argc, char *argv[])

{
if (argc != 2)
{

	printf("\nLe programme doit recevoir un et un seul argument\n");
	return(EXIT_FAILURE);

}

int N ;
double temps_final;
double omega ;
double y0 ;
double yT;


FILE* fichier = NULL;
fichier = fopen(argv[1],"r");


if (fichier == NULL)
{
	printf("Impossible d’ouvrir le fichier %s \n",argv[1]);
	return(EXIT_FAILURE);
}


fscanf(fichier, "%d %lf %lf %lf %lf", &N , &temps_final, &omega , &y0 , &yT);


printf("Nombre de pas de temps: %d \n",N);
printf("Temps final: %f \n",temps_final);
printf("Periode de la sinusoide: %f \n",omega);
printf("Condition initiale: %f \n",y0);
printf("Condition finale: %f \n",yT);

double* ynimp = NULL;
double dt;

dt = temps_final/N;

printf("Pas de temps %f \n",dt);

ynimp = tir_euler_imp(omega,dt,y0,yT,N);


free(ynimp);
fclose(fichier);
return 0;
}