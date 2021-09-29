function [q,me]=unitsquaremesh(Nx,Ny)
  
% je vais faire bloc par bloc

q=[]
me=[]



for k=0:Ny-1 
  
  q=[q bloc(k,Nx,Ny)]
  
end

%je fais ca ligne par ligne


for i=1:Ny-1
  
  %je cree ma matrice ligne 
  ligne=[i*Nx+1:(i+1)*Nx;(i-1)*Nx+1:i*Nx]
  
  %triangles du bas
  
  for j=1:Nx-1
    me=[me; ligne(2,j) ligne(2,j+1), ligne(1,j)]
  end

  %trianges du haut

  for j=1:Nx-1
    me=[me; ligne(1,j) ligne(1,j+1) ligne(2,j+1)]
  end
  
  
end  

me=me'
  
   
endfunction
