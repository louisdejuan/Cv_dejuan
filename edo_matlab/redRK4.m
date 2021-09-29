function [ t ,Y]=redRK4(f,a,b,yo,N)
  

  yn=[yo];
  h=(b-a)/N;
  t=a:h:b;
  
  for i=1:N
    
    
    k1=f(t(i),yn(i));
    k2=f(t(i)+h/2,yn(i)+(h/2)*k1);
    k3=f(t(i)+h/2,yn(i)+(h/2)*k2);
    k4=f(t(i)+h,yn(i)+h*k3);
    
    yn=[yn [yn(i)+(h/6)*(k1+2*k2+2*k3+k4)]];
    
    
    
    
  endfor
    
    Y=yn;
  
  endfunction