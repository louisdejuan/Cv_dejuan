function SysSol=SysSolData()
% SysSol : structure de donnees contenant les positions et vitesses
% de planetes (Soleils, Jupiter, Saturne, Uranus, Nepture, Pluton,
% Mercure, Terre, Lune, Mars, Venus) le 06 mai 2019 a 00h00'00''
% Le champ masse est la masse en kg et le champ  masseUA  en U.A.
% Les champs color et rayon sont utilises pour la representation graphique
% et ne correspondent pas a des parametres physiques.
% Cette structure a ete generee a partir des donnees obtenu sur le site
% "JPL Solar System Dynamics", http://ssd.jpl.nasa.gov


%SysSol.G = 2.95912208286*10^-4;
SysSol.G = 4*pi^2/(365.242198 )^2;
SysSol.Date=[2019 5 06 0 0 0];%'2019-May-06 00:00:00.0000';

colors=[ ...
         0    1.0000    0.6207; ...
         0    0.5172    1.0000; ...
    0.6897    0.1034    0.3793; ...
    0.2759    0.4138         0; ...
    1.0000    0.6552    0.3103; ...
    0.1379         0    0.3448; ...
         0    1.0000         0; ...
    1.0000    0.6552    1.0000];

R=0.3;

planete.nom='Soleil';
planete.masse=1.9891*10^30;
planete.position=[0;0;0];
planete.vitesse=[0;0;0];
planete.color=[1 1 0]; % yellow
planete.rayon=R;
planete.orbital_periode=0; %in day

i=2;
planete(i).nom='Jupiter';
planete(i).masse=1898.13*10^24;
planete(i).position=[-1.251860974202657E+00;-5.161328862554263E+00;4.944799344074653E-02];
planete(i).vitesse=[7.249979907733473E-03;-1.424673010401595E-03;-1.563058032083005E-04];
planete(i).color=colors(i-1,:);
planete(i).rayon=R/2;
planete(i).orbital_periode=4332.589; %in day

i=3;
planete(i).nom='Saturne';
planete(i).masse=5.6834*10^26;
planete(i).position=[2.601187577991443E+00;-9.711259652224602E+00;6.527935978961320E-02];
planete(i).vitesse=[5.089686376176589E-03;1.426195828017041E-03;-2.276126188618417E-04];
planete(i).color=colors(i-1,:);
planete(i).rayon=R/2;
planete(i).orbital_periode=10755.698; %in day

i=4;
planete(i).nom='Uranus';
planete(i).masse=86.813*10^24;
planete(i).position=[1.675477750770842E+01; 1.063807248535963E+01;-1.774680263140523E-01];
planete(i).vitesse=[-2.129983803522142E-03; 3.135642697975387E-03; 3.921815552265899E-05];
planete(i).color=colors(i-1,:);
planete(i).rayon=R/2;
planete(i).orbital_periode=30685.4; %in day

i=5;
planete(i).nom='Neptune';
planete(i).masse=102.413*10^24;
planete(i).position=[2.907637575537439E+01;-7.105902530444829E+00;-5.238466761362451E-01];
planete(i).vitesse=[7.315807478092233E-04; 3.067345234448702E-03;-7.993345456728092E-05];
planete(i).color=colors(i-1,:);
planete(i).rayon=R/2;
planete(i).orbital_periode=60189; %in day

i=6;
planete(i).nom='Pluton';
planete(i).masse=1.307*10^22;
planete(i).position=[1.225861987215159E+01;-3.149104050084050E+01;-1.764467636218660E-01];
planete(i).vitesse=[3.011244937957621E-03; 4.746866046954070E-04;-9.289909393732918E-04];
planete(i).color=colors(i-1,:);
planete(i).rayon=R/2;
planete(i).orbital_periode=249.58932*365.25636042 ; %in day 

i=7;
planete(i).nom='Mercure';
planete(i).masse=3.302*10^23;
planete(i).position=[3.548297835020574E-01;-1.359327500830998E-01;-4.365852873793565E-02];
planete(i).vitesse=[4.583891222955171E-03; 2.754184969437564E-02; 1.830038560391851E-03];
planete(i).color=colors(i-1,:);
planete(i).rayon=R/4;
planete(i).orbital_periode=87.969257; %in day

i=8;
planete(i).nom='Terre';
planete(i).masse=5.97219*10^24;
planete(i).position=[-7.139894133710962E-01;-7.124861956529389E-01; 3.655431990332799E-05];
planete(i).vitesse=[ 1.187851802544477E-02;-1.224668370651942E-02; 1.123931305118045E-07];
planete(i).color=[0 0 1];
planete(i).rayon=R/4;
planete(i).orbital_periode=365.25636; %in day

i=9;
planete(i).nom='Lune';
planete(i).masse=734.9*10^20;
planete(i).position=[-7.126196661128630E-01;-7.103194713354788E-01;-1.413003285202208E-04];
planete(i).vitesse=[ 1.136662981123423E-02;-1.194853031305502E-02; 3.409024779531338E-05];
planete(i).color=colors(i-2,:);
planete(i).rayon=R/20;
planete(i).orbital_periode=27.321582; % in day

i=10;
planete(i).nom='Mars';
planete(i).masse=6.4171*10^23;
planete(i).position=[-4.441162811575731E-01; 1.543417741162161E+00; 4.323714110101488E-02];
planete(i).vitesse=[-1.291934330596041E-02;-2.680101424954166E-03; 2.608393997801367E-04];
planete(i).color=[1 0 0];
planete(i).rayon=R/4;
planete(i).orbital_periode=686.98; % in day

i=11;
planete(i).nom='Venus';
planete(i).masse=48.685*10^23;
planete(i).position=[6.818951444267687E-01;-2.502090639583593E-01;-4.278367233487078E-02];
planete(i).vitesse=[6.857701661904744E-03; 1.890069084740143E-02;-1.363992918977883E-04];
planete(i).color=colors(i-3,:);
planete(i).rayon=R/4;
planete(i).orbital_periode=224.70079922; % in day

N=length(planete);
planete(1).masseUA=1.00000597682;
for i=2:N
   planete(i).masseUA=planete(i).masse/planete(1).masse*planete(1).masseUA;
end
SysSol.planetes=planete;

