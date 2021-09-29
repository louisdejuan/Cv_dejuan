function [ba]=barycentre(q,me)
  
  nme=size(me,2)
  ba=zeros(2,nme)
  
  
  for (k=1:nme)
    
    
    ba(:,k)=mean(q(:,me(:,k)'),2);
    
    
    
  end 
  
  
  
  
  
  
  
endfunction
