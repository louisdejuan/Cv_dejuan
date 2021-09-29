#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "newton.h"


double newton(double xk, double xk_1, double xk_2, double b, double eps,int kmax,int cas)

{

if ( cas==1 )   {

double tol=eps+1;
int k=1;
double xcopie;

while ((tol > eps) && (k<kmax))
{
	xcopie=xk;
	xk=xk-((xk-b*sin(xk)-2.*xk_1+xk_2)/(1.-b*cos(xk)));

	tol=fabs(xk-xcopie);
	printf("k%d tol%lf  \n",k,tol);
	k=k+1;


}

printf("xk=%lf  tol=%lf k=%d \n",xk,tol,k);



}

else if (cas == 2 ) {



double tol=eps+1;
double k=1;	
double xcopie;


while ((tol>eps) && (k<kmax)) 
{
	xcopie=xk;
	xk=xk-(((-b/4.)*(sin(xk)+2.*sin(xk_1)+sin(xk_2)))-2.*xk_1+xk_2+xk)/(1.-(-b/4.)*cos(xk));
	tol=fabs(xk-xcopie);
	k=k+1;
}




}

return (xk);


}
