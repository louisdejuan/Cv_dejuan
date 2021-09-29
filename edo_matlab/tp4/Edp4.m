function [Y,erreur,h]=Edp4(n)

a=0;
b=2*pi;
nu=0.001;

position_initiale=1;

position_finale=4*pi+1+2*nu;


f=@(t) cos(t);

h=(b-a)/n;
i=1:1:n-1;
x=a+i.*h;

Y=[];
L=(-1)*Lap1d(n-1);
L(n-1,n-1)=(2-(nu/(h+nu)));


vectf=f(x)*(h^2);
vectf(1)=vectf(1)+position_initiale;
vectf(n-1)=vectf(n-1)+position_finale*(h/(h+nu));

Y=L\vectf';


g=@(t) cos(x)+2*x;
Y_reel=g(x);

erreur=max(abs(Y'-Y_reel));




  
    


  
  
  
  
endfunction