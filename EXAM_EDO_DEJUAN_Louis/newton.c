
#include "newton.h"


double newton(int nb_it_max, double tolerance, double itere_initial,double A, double B,double (*fonc) (double, double, double), double (*deriv_fonc) (double, double))


{


double itere_courant = itere_initial;
double test = tolerance+1.;

int k=0;
double itere_copie;



while ((test > tolerance) && (k<nb_it_max))
{
	
	itere_copie=itere_courant;
	
	itere_courant=itere_courant-((*fonc) (itere_courant,A,B)/(*deriv_fonc) (itere_courant,A));

	test=fabs(itere_courant-itere_copie);

	
	k=k+1;


}





return itere_courant; // resultat de la proc ́edure



}
