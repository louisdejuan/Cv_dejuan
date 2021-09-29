function N=fonction_norme(i,j,Matrice_y)
  
  
  N=power(sqrt(power(Matrice_y(1,i)-Matrice_y(1,j),2)+power(Matrice_y(2,i)-Matrice_y(2,j),2)+power(Matrice_y(3,i)-Matrice_y(3,j),2)),3);
  
  
  
endfunction
