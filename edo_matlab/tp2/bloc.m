function [q]=bloc(k,Nx,Ny)
  
  q=zeros(2,Nx);
  
  
  %ligne d'abscisse
  
  abscisse=zeros(1,Nx)
  for i=0:Nx-1
    abscisse(i+1)=i*(1/(Nx-1))
  end
  
  %ligne d'ordonnee
  point_y=k*(1/(Ny-1));
  ordonnee=ones(1,Nx)*point_y;
  
  
  
  q(1,:)=abscisse
  q(2,:)=ordonnee
  
  
endfunction
