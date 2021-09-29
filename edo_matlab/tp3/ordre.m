a=0
b=10
y0=exp(1)

f=@(t,y) -y.*sin(t);
g=@(t) exp(cos(t));


iteration=100:100:1000;
h=[];



EUP_erreur=[]
TGA_erreur=[]
RK4_erreur=[]
AB3_erreur=[]
PC4_erreur=[]

for N=iteration
  

h=[h (b-a)/N];


 
  
  
[t,Y_EUP]=redEUP(f,a,b,y0,N);

[t,Y_TGA]=redTGA(f,a,b,y0,N);

[t,Y_RK4]=redRK4(f,a,b,y0,N);

[t,Y_AB3]=redAB3(f,a,b,y0,N);

[t,Y_PC4]=redPC4(f,a,b,y0,N);

Y_reelle=g(t); 


EUP_erreur=[EUP_erreur [max([abs(Y_EUP-Y_reelle)])]];

TGA_erreur=[TGA_erreur [max([abs(Y_TGA-Y_reelle)])]];

RK4_erreur=[RK4_erreur [max([abs(Y_RK4-Y_reelle)])]];

AB3_erreur=[AB3_erreur [max([abs(Y_AB3-Y_reelle)])]];

PC4_erreur=[PC4_erreur [max([abs(Y_PC4-Y_reelle)])]];

endfor



plot(h,EUP_erreur)
hold on;
plot(h,TGA_erreur)
hold on;
plot(h,RK4_erreur)
hold on;
plot(h,AB3_erreur)
hold on;
plot(h,PC4_erreur)




