function x=pb_N_corps(N_planetes,N_iterations,T)
  
  SysSolN=SysSolData();
  SysSolN.planetes=SysSolN.planetes(1:N_planetes);
  
  Matrice_masse=[];
  Matrice_position=[];
  Matrice_vitesse=[];
  
  
 
 t=0;
  
  
  for l=1:N_planetes
    
    Matrice_masse=[Matrice_masse [SysSolN.planetes(l).masse]];
    Matrice_position=[Matrice_position [SysSolN.planetes(l).position]];
    Matrice_vitesse=[Matrice_vitesse [SysSolN.planetes(l).vitesse]];
    
    
    
  endfor
  
  Matrice_masse=Matrice_masse/Matrice_masse(1);
  Matrice_y=[Matrice_position Matrice_vitesse];
  Matrice_y_finale=Matrice_y;
  
  
  h=T/N_iterations;
  for k=1:N_iterations
    
    disp(k);
   
    Matrice_y +=h*fonction_calcul(Matrice_y,t,N_planetes,Matrice_masse);
    
    Matrice_y_finale=[Matrice_y_finale ; Matrice_y];
    
    
  endfor
  
  R=Matrice_y_finale;
  
  vect_position=R(:,1:N_planetes);
  
  
  
  A=zeros(N_planetes,3,N_iterations+1);
  
  

    
    
      for k=0:N_iterations
    
      A(:,:,k+1)=vect_position(3*(k)+1:3*(k)+3,:)';
    
    end  
    
    
    x=A;
    
    
    
   
  
  
  
  
  
  
  
  
endfunction
