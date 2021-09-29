function Y=fonction_calcul(Matrice_y,t,N_planetes,Matrice_masse)
  
  Gx=Gy=Gz=Y=Yx=Yy=Yz=[];
  C_G=4*pi^2/(365.242198 )^2;
  
  
  Ax=Ay=Az=zeros(N_planetes);
  
  for i=1:N_planetes
    for j=1:N_planetes
      
      
      norme=fonction_norme(i,j,Matrice_y);
      
      
      if (i==j)
        
        somme=fonction_somme(Matrice_y,Matrice_masse,i,N_planetes);
        
        Ax(i,j)=-C_G*Matrice_y(1,i)*somme;
        Ay(i,j)=-C_G*Matrice_y(2,i)*somme;
        Az(i,j)=-C_G*Matrice_y(3,i)*somme;
      end
      
        
      if(i~=j)
        if (norme==0) disp('erreur') ; end
        Ax(i,j)=C_G*Matrice_masse(j)*(1/norme);
        Ay(i,j)=C_G*Matrice_masse(j)*(1/norme);
        Az(i,j)=C_G*Matrice_masse(j)*(1/norme);
        
      end
    end
   end
   
   
   Gx=[[zeros(N_planetes)] [eye(N_planetes)];[Ax] [zeros(N_planetes)]];
   Gy=[[zeros(N_planetes)] [eye(N_planetes)];[Ay] [zeros(N_planetes)]];
   Gz=[[zeros(N_planetes)] [eye(N_planetes)];[Az] [zeros(N_planetes)]];
   
  
   
   Yx=Gx*Matrice_y(1,:)';
   Yy=Gy*Matrice_y(2,:)';
   Yz=Gz*Matrice_y(3,:)';
   
   
   
   
   Y=[Yx';Yy';Yz'];
   
   
endfunction
