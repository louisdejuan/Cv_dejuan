n=3000
a=-1
b=pi

iteration=100:100:n;

f=@(x) -4.*cos(2*x-1)+9*sin(3*x)
g=@(x) -cos(2*x-1)+sin(3*x)

ua=g(a)
ub=g(b)

Y_reelle=Y_der=der1_erreur=h=[];
for N=iteration
  
  disp(N)

h=[h (b-a)/N];

h_ite=(b-a)/N;
i=0:1:N-1;
x=a+i.*h_ite;

Y_reelle=g(x);

Mat=Lap1d(N);


vectf=f(x);
vectf(1)=vectf(1)+ua/(h_ite^2);
vectf(N)=vectf(N)+ub/(h_ite^2);
Y_der=-(Mat./(h_ite^2))\vectf';


der1_erreur=[der1_erreur [max([abs(Y_der'-Y_reelle)])]];





end

loglog(h,der1_erreur)
hold on;