function der=derive1(a,b,n,f)
  
  
  h=(b-a)/n;
  i=0:1:n;
  x=a+i.*h;
  
  der=[];
  der=(f(x.+h)-f(x))./h;
  
  
  
  
  
  
endfunction
