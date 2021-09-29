a=0
b=10
y0=exp(1)

N=100

f=@(t,y) -y.*sin(t);
g=@(t) exp(cos(t));

[t,Y_EUP]=redEUP(f,a,b,y0,N);

[t,Y_TGA]=redTGA(f,a,b,y0,N);

[t,Y_RK4]=redRK4(f,a,b,y0,N);

[t,Y_AB3]=redAB3(f,a,b,y0,N);

[t,Y_PC4]=redPC4(f,a,b,y0,N);

Y_reelle=g(t);

EUP_erreur=[abs(Y_EUP-Y_reelle)];

TGA_erreur=[abs(Y_TGA-Y_reelle)];

RK4_erreur=[abs(Y_RK4-Y_reelle)];

AB3_erreur=[abs(Y_AB3-Y_reelle)];

PC4_erreur=[abs(Y_PC4-Y_reelle)];


eup=figure;
plot(t,EUP_erreur)

tga=figure;
plot(t,TGA_erreur)

rk4=figure;
plot(t,RK4_erreur)

ab3=figure;
plot(t,AB3_erreur)

pc4=figure;
plot(t,PC4_erreur)













