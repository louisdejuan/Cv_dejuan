function [t,Y]=redTGA(f,a,b,yo,N)
  
  yn=[yo];
  h=(b-a)/N;
  t=a:h:b;
  
  for i=1:N
    
    
    
    yn=[yn [yn(i) + h*f(t(i)+h/2,yn(i)+(h/2)*f(t(i),yn(i)))] ];
    
  end
  
  
  Y=yn;
  
  
  
endfunction
