function [Y,erreur,h]=Edp3(n)

a=0;
b=2*pi;
vitesse_initiale=2;
position_finale=4*pi+1;


f=@(t) cos(t);

h=(b-a)/n;
i=1:1:n-1;
x=a+i.*h;

Y=[];
L=(-1)*Lap1d(n-1);
L(1,1)=1;

vectf=f(x)*(h^2);
vectf(1)=(3/2)*vectf(1)-vitesse_initiale*h;
vectf(n-1)=vectf(n-1)+position_finale;

Y=L\vectf';


g=@(t) cos(x)+2*x;
Y_reel=g(x);

erreur=max(abs(Y'-Y_reel));






  
    


  
  
  
  
endfunction