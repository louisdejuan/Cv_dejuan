function [t,Y]=redAB3(f,a,b,yo,N) 
  
  [t,Y_test]=redRK4(f,a,b,yo,N);
  yn=[Y_test(1) Y_test(2) Y_test(3)];
  h=(b-a)/N;
  
  
  for i=3:N
    
    yn=[yn [yn(i)+(h/12)*(23*f(t(i),yn(i))-16*f(t(i-1),yn(i-1))+5*f(t(i-2),yn(i-2)))]];
    
  end  
  
  Y=yn;
  
  
endfunction
