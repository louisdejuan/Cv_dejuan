function plotSysSol(t,Y,SysSol)
% Representation des orbites des planetes de  la structure SysSol (voir SysSolData.m ou
% SysSolDataHairer.m).
% Les positions successives de la planete i sont donnees par
%   x=Y(3*(i-1)+1,:), y=Y(3*(i-1)+2,:) et z=Y(3*(i-1)+3,:)
% Les temps successifs t ne sont pas utiles (pour le moment)
N=length(SysSol.planetes);
k=1;
hold on
for i=1:N
    plot3(Y(k,:),Y(k+1,:),Y(k+2,:),'Color',SysSol.planetes(i).color);
    k=k+3;
    legendCell(i)={SysSol.planetes(i).nom};
end
log=ver;
if strcmp(log(1).Name,'Octave')
  legend(legendCell,'Location','BestOutside')
else % Matlab
  if verLessThan('matlab','9.2') % R2017a
    legend(legendCell,'Location','BestOutside')
  else 
    legend(legendCell,'Location','BestOutside','AutoUpdate','off')
  end
end
axis equal
axis off
view([0 -10])
