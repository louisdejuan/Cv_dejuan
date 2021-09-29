function PlotPlanet(Pos,Rayon,Color)
% Representation d'une planete en position Pos, de rayon Rayon
% et de couleur Color
[X,Y,Z] = sphere(10);
X=X*Rayon+Pos(1);
Y=Y*Rayon+Pos(2);
Z=Z*Rayon+Pos(3);
sph = surf(X,Y,Z,'FaceColor',Color,'EdgeColor',Color);

%shading interp
%axis equal