N=10:100:1010


list_erreur=[];
list_h=[];

for i=N
  
  
  i
  [Y,erreur,h]=Edp1(i);
  list_erreur=[list_erreur [erreur]];
  list_h=[list_h [h]];
  
    
endfor

loglog(list_h,list_erreur)


