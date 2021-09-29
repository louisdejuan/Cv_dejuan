a=0.03134
b=1.5887e-10
yo=3900000

N=1000




t1=1790
t2=2000


f=@(t,y) a*y-b*y.^2
f_reel=@(t) (a*yo)./((b*yo)+(a-b*yo).*exp(-a*(t-t1)))





Y_reel=RK4_erreur=[];
h=[];
iteration=100:100:1000

for N=iteration
  
  
h=[h (t2-t1)/N];




[t,Y_RK4]=redRK4(f,t1,t2,yo,N);
Y_reel=f_reel(t);


RK4_erreur=[RK4_erreur [max([abs(Y_RK4-Y_reel)])]];


end


loglog(h,RK4_erreur)
figure
plot(t,Y_RK4)






