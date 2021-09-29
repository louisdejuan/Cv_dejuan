function s=fonction_somme(Matrice_y,Matrice_masse,i,N_planetes)
  
  s=0;
  for j=1:N_planetes
    
    if (i==j)
      s=s+0;
    else
      
      a=fonction_norme(i,j,Matrice_y);
      s=s+(Matrice_masse(j)/a);
      
     
      
    end  
      
    
    
  end
  
  
  
  
  
endfunction
