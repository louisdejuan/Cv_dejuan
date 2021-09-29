function x=mosaique18(n)
  
   x=quadrillage(-n-1,n+1,-n-1,n+1)
   
   for i=-n:n
    
    for j=-n:n
      
      
      
      if mod(i+j,4)!=0
        
        x=black(i,j);
      
        
        
        
      
      
    end
    
   end
  
  
  
  
endfunction
