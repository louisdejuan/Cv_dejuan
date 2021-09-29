
// fichier f1.c


#include "f1.h"


double f (double temps, double y, double alpha, double y_0)

{
	return (-(alpha*pow(y,2))/((y_0+temps)*exp(-alpha*temps)))+exp(-alpha*temps);
}

double f1 (double x)


{
return pow(x,2) - 2.0;
}


double deriv_f1 (double x)

{
return 2.*x;
}