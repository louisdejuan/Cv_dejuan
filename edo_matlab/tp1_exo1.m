function [a] = tp1_exo1(A,B,C,n,x)

[xA,yA]=A
[xB,yB]=B
[xC,yC]=C



plot([xA,xB],[yA,yB]) 
plot([xB,xC],[yB,yC]) 
plot([xC,xD],[yC,yD]) 



for k = 1:n
  
  xA=x*(xB-xA)+xA
  yA=x*(yB-yA)+yA
  xB=x*(xB-xC)+xB
  yB=x*(yB-yC)+yB
  xC=x*(xC-xA)+xC
  yC=x*(yC-yA)+yC
  
  plot([xA,xB],[yA,yB]) 
  plot([xB,xC],[yB,yC]) 
  plot([xC,xD],[yC,yD])     
  
  
end

a=0


 
end
