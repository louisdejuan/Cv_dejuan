function [x] = quadrillage(imin,imax,jmin,jmax)
  
  
  
  
  
  
  for i=imin:imax
    
    plot([i,i],[jmin,jmax],'b')
    hold on;
    
    
  end
  
  for j=jmin:jmax
    
    
    plot([imin,imax],[j,j],'b')
    hold on;
    
  
  end
  
  
x=0
  
  
  
endfunction
