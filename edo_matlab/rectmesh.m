function [q,me]=rectmesh(a,b,c,d,Nx,Ny)
  
  %les points de connectivité sont les memes, on les recuperes sur la fonction precédente
  [q,me]=unitsquaremesh(Nx,Ny)
  
  
  
  q(1,:)=a+(b-a)*q(1,:)
  q(2,:)=c+(d-c)*q(2,:)
  
  
  
  
  
endfunction
