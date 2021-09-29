function Mat=Lap1d(n)
  
  
  
  Mat_remplissage_inf= Mat_remplissage_sup=zeros(n);
  Mat_remplissage_inf(2:n,1:n-1)=eye(n-1);
  Mat_remplissage_sup(1:n-1,2:n)=eye(n-1);
  
  Mat=zeros(n)-2*eye(n)+Mat_remplissage_inf+Mat_remplissage_sup;
  
  
  
  
endfunction
