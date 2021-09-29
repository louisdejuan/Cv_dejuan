function [x] = mosaique26(n)
  
  x=quadrillage(-n-1,n+1,-n-1,n+1)
  
  for i=-n:n
    
    for j=-n:n
        
        if mod(i,2)==0 && i>-j && i>j
          
           x=black(i,j)
           
        
           
        end
        
        if mod(j,2)==0 && i>-j && j>i
          
          x=black(i,j)
        end
        
        if mod(i,2)==0 && j>i && i<-j
          
          x=black(i,j)
          
        end
        
        if mod(j,2)==0 && i<-j && i>j 
          x=black(i,j)
          
        end
        
        if mod(i,2)==1 && i==j
          
          x=black(i,i)
        end
        
        
        
        
        
   
      end
        
        
      
      
      
      
    end
    
    x=0;
    
endfunction
  
  
  
  
  
  
  
  
  

