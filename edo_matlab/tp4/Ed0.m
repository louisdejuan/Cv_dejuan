
n=100
a=-1
b=pi


f=@(x) -4.*cos(2*x-1)+9*sin(3*x)
g=@(x) -cos(2*x-1)+sin(3*x)
ua=g(a)
ub=g(b)



Mat=Lap1d(n);

h=(b-a)/n;
i=0:1:n-1;
x=a+i.*h;
vectf=f(x);
vectf(1)=vectf(1)+ua/(h^2)
vectf(n)=vectf(n)+ub/(h^2)




u=-(Mat./(h^2))\vectf';


plot(x,u')



