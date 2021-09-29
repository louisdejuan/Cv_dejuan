function der=derive2(a,b,n,f)
  
  h=(b-a)/n;
  i=0:1:n;
  x=a+i.*h;
  
  der=[];
  der=-(f(x+2*h)-4*f(x+h)+3*f(x))./(2*h);
  
  
  
  
endfunction
