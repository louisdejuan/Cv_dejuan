function der=derive_ordre2(a,b,n,f)
  
  h=(b-a)/n;
  i=0:1:n;
  x=a+i.*h;
  
  der=[];
  der=(f(x+h)-2*f(x)+f(x-h))./(h^2);
  
  
  
  
endfunction
