#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "euler_imp.h"
#include "tir_euler_imp.h"
double* tir_euler_imp(double omega, double pas_de_temps, double cond_init,

double cond_finale, int nombre_de_pas)



{

double val1;

double val2;
double epsilon = 0.001;
double aa;
double bb;
double eul1, eul2;
double* valeur = NULL;



/** premier essai **/
val1=cond_init*(1+epsilon);



valeur = euler_imp(omega, pas_de_temps, cond_init, val1, nombre_de_pas);

eul1 = valeur[nombre_de_pas-1];

free(valeur); // on lib`ere la m ́emoire allou ́ee dans euler_imp
valeur=NULL;





/** deuxi`eme essai **/
val2=cond_init*(1-epsilon);


valeur = euler_imp(omega, pas_de_temps, cond_init, val2, nombre_de_pas);

eul2 = valeur[nombre_de_pas-1];



free(valeur); // on lib`ere la m ́emoire allou ́ee dans euler_imp

valeur=NULL;


printf("check val1 %lf check val2 %lf \n",val1,val2);
printf("check  eul1 %lf check eul2 %lf \n",eul1,eul2);

aa = (eul2 - eul1)/(val2-val1);
bb = eul1 - aa*val1;

printf("check alpha %lf check beta %lf \n",aa,bb);

/** valeur de y1 et verif **/

val1 = (cond_finale - bb)/aa;



valeur = euler_imp(omega, pas_de_temps, cond_init, val1, nombre_de_pas);

printf("check %lf \n ",valeur[1]);

printf("verification %lf %lf \n",valeur[nombre_de_pas-1],cond_finale);



return(valeur); // le programme principal pourra utiliser tout le tableau


}