
#include "newton.h"


double newton(int nb_it_max, double tolerance, double itere_initial,double (*fonc) (double), double (*deriv_fonc) (double))


{


double itere_courant = itere_initial;
double test = tolerance+1;

int k=0;
double itere_copie;



while ((test > tolerance) && (k<nb_it_max))
{
	
	itere_copie=itere_courant;
	itere_courant=itere_courant-((*fonc) (itere_courant)/(*deriv_fonc) (itere_courant));

	test=fabs(itere_courant-itere_copie);

	
	k=k+1;


}



printf("residu %lf \n",(*fonc) (itere_courant));

return itere_courant; // resultat de la proc ́edure



}
