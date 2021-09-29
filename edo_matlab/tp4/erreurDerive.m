a=0
b=10
n=10000


iteration=100:100:n;

f=@(t) t.^4;
g=@(t) 4.*t.^3




Y_der1=Y_der2=der1_erreur=der2_erreur=h=[];

for N=iteration
  
  disp(N)

h=[h (b-a)/N];

h_ite=(b-a)/N;
i=0:1:N;
x=a+i.*h_ite;

Y_reelle=g(x);


Y_der1=derive1(a,b,N,f); 
  
der1_erreur=[der1_erreur [max([abs(Y_der1-Y_reelle)])]];

Y_der2=derive2(a,b,N,f); 
  
der2_erreur=[der2_erreur [max([abs(Y_der2-Y_reelle)])]];


end


loglog(h,der1_erreur)
hold on;

loglog(h,der2_erreur)
hold on;






