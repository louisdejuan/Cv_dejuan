function [Y,erreur,h]=Edp5(n)

a=0;
b=2*pi;
nu=1;

position_initiale=1;

position_finale=4*pi+1+2*nu;


f=@(t) cos(t);

h=(b-a)/n;
i=1:1:n-1;
x=a+i.*h;
  
Y=[];
L=(-1)*Lap1d(n-1);
L(n-1,n-1)=2;
L(n-1,n-2)=-(1+(nu/(2*h+nu)));


vectf=f(x)*(h^2);
vectf(1)=vectf(1)+position_initiale;
vectf(n-1)=vectf(n-1)+position_finale*((2*h)/(2*h+nu));


Y=L\vectf';


g=@(t) cos(x)+2*x;
Y_reel=g(x);

erreur=max(abs(Y'-Y_reel));






  
    


  
  
  
  
endfunction