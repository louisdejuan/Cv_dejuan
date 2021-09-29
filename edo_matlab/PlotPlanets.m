function PlotPlanets(SysSol,Q)
% Representation des planetes de la structure SysSol (voir SysSolData.m ou
% SysSolDataHairer.m) en positions Q :
% La planete 1 en position Q(1:3)
% ...
% La planete i en position Q(3*(i-1)+:3*i)
% ...
hold on

N=length(SysSol.planetes);
for i=1:N
    PlotPlanet(Q((3*(i-1)+1):3*i),SysSol.planetes(i).rayon,SysSol.planetes(i).color)
    %pause
end
hold off