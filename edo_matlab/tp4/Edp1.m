function [Y,erreur,h]=Edp1(n)
  
  
  
  a=0;
  b=1;
  
 
  
  nu=4;
  condition_a=0;
  condition_b=0
  
  
  
  f=@(t) t;
  
  
  h=(b-a)/n;
  i=1:1:n-1;
  x=a+i.*h;
  
  
  
  Y=[];
  D1=(-1)*Lap1d(n-1);
  L=D1+(h^2)*nu*eye(n-1);
  vectf=(h^2)*f(x);
  vectf(1)=vectf(1)+condition_a;
  vectf(n-1)=vectf(n-1)+condition_b;
  size(vectf);
  size(x);
  size(L);
  
  
  Y=L\vectf'
  
  
  
  
  C1=0.25/(exp(2)-exp(-2));
  C2=-C1;
  
  g=@(t) C1*exp(-2.*t)+C2*exp(2.*t)+(t/4);
  
  y_reel=g(x)
  
  
  erreur=max(abs(Y'-y_reel))
  
  
  
  
  
  
  
  
  
  
  
  
endfunction
