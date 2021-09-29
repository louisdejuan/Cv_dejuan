function [t,Y]=redPC4(f,a,b,yo,N)
  
  [t,Y_test]=redRK4(f,a,b,yo,N);
  yn=[Y_test(1) Y_test(2) Y_test(3) Y_test(4)];
  h=(b-a)/N;
  

  
  for i=4:N
    
    yn_barre=yn(i)+(h/24)*(55*f(t(i),yn(i))-59*f(t(i-1),yn(i-1))+37*f(t(i-2),yn(i-2))-9*f(t(i-3),yn(i-3)));
    yn=[yn [yn(i)+(h/24)*(9*f(t(i),yn_barre)+19*f(t(i),yn(i))-5*f(t(i),yn(i-1))+f(t(i-1),yn(i-2)))]];
    
    
    
  end
  
  Y=yn;
  
  
  
endfunction
