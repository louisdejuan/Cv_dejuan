#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 10:25:54 2021

@author: louis
"""
import numpy as np
import matplotlib.pyplot as plt




def  Simulerbrownien(T,N):
    delta_t = T/N
    ecart_type = np.sqrt(delta_t)
    normal = np.random.normal(loc=0, scale = ecart_type, size = N)
    
# Simulation d'un movement brownien standard
# On remaet le W_0 = 0 pour assurer le mouvement initial
    normal[0]=0.
    brownien_standard= np.cumsum(normal)
    return brownien_standard

 
    
def Simulerbrownien_montecarlo(T,N,Nb_montecarlo):
    '''
        on simule un nombre fini de brownien stockés dans un tableau ou les lignes correspondent à des trajectoires de montecarlo
    '''
    delta_t = T/N
    ecart_type = np.sqrt(delta_t)
    normal = np.random.normal(loc=0, scale = ecart_type, size = (Nb_montecarlo,N))
    normal[:,0]=0.
    #on additionne sur par ligne pour toutes les colonnes
    result=np.cumsum(normal,1)
    return result

def trajectoire_actif_montecarlo(s0,r,sigma,period,brw_montecarlo):
    '''
        on simule toutes les trajectoires (St) de l'actif risqué en fonction du brownien

    '''
    
    drift = (r-(sigma**2)/2.)*period
    
    return s0*np.exp(drift + sigma*brw_montecarlo)


def prix_barriere(K,B,r,T,St):
    '''
        on calcule le prix barriere et l'intervalle de confiance en fonction des trajectoires simuléés écrites précédemment
    '''
    max_St = np.max(St,axis=1)
    actualisation = np.exp(-r*T)
    payoffs = np.maximum(St[:,-1]-K,0)* (max_St > B ) * actualisation 
    
    prix_barr = np.mean(payoffs)
    erreur_barriere = 1.96 * np.std(payoffs) / np.sqrt(payoffs.shape[0])
    
    interval_confiance = [prix_barr-erreur_barriere, prix_barr+erreur_barriere]
    
    return prix_barr, interval_confiance

def prix_barriere_maturite(K,B,r,T,St):
    
     Nb_simul,N=St.shape
     period = np.linspace(start = 0,stop = T, num = N, endpoint = True)
     actualisation=np.exp(-r*period)
     #calcul du max de Su pour u < t pour tout t entre [0 T]
     max_St=np.maximum.accumulate(St,axis=1)
     #calcul du payoff actualisé a tout instant
     payoffs = np.maximum(St-K,0)* (max_St > B ) * actualisation 
     
     
     prix_barr_maturite=np.mean(payoffs, axis=0)
     #intervalle de confiance
     erreur_prix=np.std(payoffs,axis=0)
     
     IC_sup=prix_barr_maturite+1.96*erreur_prix/np.sqrt(Nb_simul)
     IC_inf=prix_barr_maturite-1.96*erreur_prix/np.sqrt(Nb_simul)
          
     return prix_barr_maturite,IC_sup,IC_inf
 

def prix_barriere_antithetique(K,B,r,T,s0,sigma,period,Wt):
    
    #on va calculer le prix pour st et le prix pour -W et W
    St1=trajectoire_actif_montecarlo(s0,r,sigma,period,Wt)
    max_St = np.max(St1,axis=1)
    actualisation = np.exp(-r*T)
    payoffs1 = np.maximum(St1[:,-1]-K,0)* (max_St > B ) * actualisation 
    
    St2=trajectoire_actif_montecarlo(s0,r,sigma,period,-Wt)
    max_St = np.max(St2,axis=1)
    
    payoffs2 = np.maximum(St2[:,-1]-K,0)* (max_St > B ) * actualisation 

    #coefficient de correlation
    correl=np.corrcoef(payoffs1,payoffs2)
    print("coef de correlation = ",correl)
    
    payoffs=(payoffs1+payoffs2)/2.
    
    prix_barr = np.mean(payoffs)
    erreur_barriere = 1.96 * np.std(payoffs) / np.sqrt(payoffs.shape[0])
    
    interval_confiance = [prix_barr-erreur_barriere, prix_barr+erreur_barriere]
    
    return prix_barr, interval_confiance


def trajectoire_moyenne_trapeze(St): 
    '''
        Fonction qui simule  des trajectoires de moyenne A_t = 1/t \int_t^T Su du
        avec A_0 = S_0
        En entree St : Une matrice(i,j) ou 
        i est l'indice des trajectoire
        j est l'indice du temps (i=0, i=N+1)
    '''
    m,n = St.shape
    St2 = (St[:,0:-1] + St[:,1:])/2.
# On remet le prix des actifs a t=0
    S_0 = np.reshape(St[:,0], newshape=(-1,1))
    St2 = np.concatenate((S_0, St2), axis = 1)
    
    return np.add.accumulate(St2,axis=1)/np.arange(1,n+1)
    
    
def trajectoire_moyenne_euler(St): 
    '''
        Fonction qui simule  des trajectoires de moyenne A_t = 1/t \int_t^T Su du
        avec A_0 = S_0
        En entree St : Une matrice(i,j) ou 
        i est l'indice des trajectoire
        j est l'indice du temps (i=0, i=N+1)
    '''
    m,n = St.shape
    return np.add.accumulate(St,axis=1)/np.arange(1,n+1)
    
def prix_asiatique_maturite(K,B,r,T,St,type = "Euler"):
    
     Nb_simul,N=St.shape
     period = np.linspace(start = 0,stop = T, num = N, endpoint = True)
     actualisation=np.exp(-r*period)
     #calcul du max de Su pour u < t pour tout t entre [0 T]
     max_St=np.maximum.accumulate(St,axis=1)
     #approximation de l'integrale par la moyenne des St entre O et T
     #calcul du payoff actualisé a tout instant
     if type == "Euler":
        estim_integrale_St = trajectoire_moyenne_euler(St)
     else:
        estim_integrale_St = trajectoire_moyenne_trapeze(St)
     
     payoffs = np.maximum(estim_integrale_St-K,0)* (max_St > B ) * actualisation 
     
     
     prix_barr_maturite=np.mean(payoffs, axis=0)
     #intervalle de confiance
     erreur_prix=np.std(payoffs,axis=0)
     
     IC_sup=prix_barr_maturite+1.96*erreur_prix/np.sqrt(Nb_simul)
     IC_inf=prix_barr_maturite-1.96*erreur_prix/np.sqrt(Nb_simul)
          
     return prix_barr_maturite,IC_sup,IC_inf
    

def etude_convergence_asiatique(r,T,s0,K,B,Ns,Ms):
    '''
        FOnction d'etude de la convergence pour les options asiatique à T=5 
        en fonction d'un vecteur de Ns (nombre de pas) et Ms (nombre de Simulations)
    '''
    # T = 5
    # N = 100000
    # nb_montecarlo = 1000
    # s0=100
    # sigma=0.2
    # r=0.05
    # K=95
    # B=105
    
# # vecteur temps
    result = []
    for n in Ns:
        period = np.linspace(start = 0,stop = T, num = n, endpoint = True)
        for m in Ms:
            print ('simulation avec : n=%s, m=%s'%(n,m))
            browniens = Simulerbrownien_montecarlo(T, n, m)
            prix_actifs = trajectoire_actif_montecarlo(s0, r, sigma, period, browniens)
            prix_asiat_maturite,icmax,icmin=prix_asiatique_maturite(K, B, r, T, prix_actifs, "Trapeze")
        
            result.append([n,m,prix_asiat_maturite[-1], icmax[-1], icmin[-1]])
       
    return result
   
def trajectoire_exo2(lamb,mu,N_simul,T,N,normal):
    delta_t = T/N
    result=np.zeros(shape=(N_simul,N))
    
    for i in range(N_simul):
        for j in range (1,N):
            X=result[i,j-1]
            result[i,j]=X+delta_t*(-lamb * (X>0) +mu *(X<0))+normal[i,j]
    return result

def probaetlog(Xt):
    proba = np.sum(np.amax(np.abs(Xt),axis=1 )<=1 )/Xt.shape[0]
    logproba = -np.log(proba)
    
    return proba,logproba
    


    
        

if __name__ == '__main__':
    
#exercice 1
    T = 5
    N = 1000
    nb_montecarlo = 1000
    s0=100.
    sigma=0.2
    r=0.05
    K=95.
    B=105.
    
# # vecteur temps
    period = np.linspace(start = 0,stop = T, num = N, endpoint = True)

# # question 2.a simuler un brownien :    
    
    brownien_standard=Simulerbrownien(T,N)


# # Quelques graphiques
    fig,ax = plt.subplots()
    ax.plot(period,brownien_standard)
    ax.grid()
    ax.set_title('Mouvement Bronwien')
    ax.set_xlabel('temps')   
    fig.savefig('brownien_standard.png') 
    
# # question 2.b
    browniens = Simulerbrownien_montecarlo(T, N, nb_montecarlo)
    prix_actifs = trajectoire_actif_montecarlo(s0, r, sigma, period, browniens)
    fig,ax = plt.subplots()
    ax.plot(period,prix_actifs[0,:])
    ax.grid()
    ax.set_title('prix_actif')
    ax.set_xlabel('temps')   
    fig.savefig('trajectoire_actif.png') 
    
    prix_bar,I_conf = prix_barriere(K,B,r,T,prix_actifs)
    
    print('estimation de phi_T =%s, à IC à 5 = %s'%(prix_bar,I_conf))
    
# # question 2.c
    
    prix_bar_maturite,icmax,icmin=prix_barriere_maturite(K, B, r, T, prix_actifs)
    fig,ax = plt.subplots()
    ax.plot(period,prix_bar_maturite)
    ax.plot(period,icmax,label="IC_sup",color="red")
    ax.plot(period,icmin,label="IC_min",color="red")
    ax.grid()
    ax.legend()
    ax.set_title('prix_actif_mat')
    ax.set_xlabel('temps')   
    fig.savefig('prix_barriere_maturite.png') 
    
# # question  2.d
    
    
    # #on veut tester la methode de reduction de variance en utilisant les memes browniens standard
    
    
    prix_bar_anti,I_conf_anti = prix_barriere_antithetique(K,B,r,T,s0,sigma,period,browniens)
    
    print('estimation de phi_T_antit =%s, à IC à 5 = %s'%(prix_bar_anti,I_conf_anti))
    
# # question 3
# # question 3.a 
    # # Simulation du processus moyenne par discretisation Euler
    A_t =  trajectoire_moyenne_euler(prix_actifs)
    fig,ax = plt.subplots()
    for i in range(10):
        ax.plot(period,A_t[i,:])
    ax.grid()
    ax.legend()
    ax.set_title('Simulation Moyenne Euler')
    ax.set_xlabel('temps')   
    fig.savefig('simulation_moyenne_euler.png') 
    
    # # Simulation du processus moyenne par discretisation Trapeze
    A_t =  trajectoire_moyenne_trapeze(prix_actifs)
    fig,ax = plt.subplots()
    for i in range(10):
        ax.plot(period,A_t[i,:])
    ax.grid()
    ax.legend()
    ax.set_title('Simulation Moyenne Trapèze')
    ax.set_xlabel('temps')   
    fig.savefig('simulation_moyenne_trapeze.png')

    # Simulation du prix d'une option barrière asiatique
    
    prix_asiat_maturite,icmax,icmin=prix_asiatique_maturite(K, B, r, T, prix_actifs, "Trapeze")
    fig,ax = plt.subplots()
    ax.plot(period,prix_asiat_maturite)
    ax.plot(period,icmax,label="IC_sup",color="red")
    ax.plot(period,icmin,label="IC_min",color="red")
    ax.grid()
    ax.legend()
    ax.set_title("prix asiatique barrière (Méthode des trapèzes)")
    ax.set_xlabel('temps')   
    fig.savefig('prix_asiatique_maturite_trapezes_2sigma.png') 
    
    #le prix asiatique diminue car la volatilité de la moyenne est plus faible
    
    Ns = np.logspace(start=1, stop=4, num=4, dtype = np.int)
    Ms = np.logspace(start=2, stop=4, num=3, dtype = np.int)
    print (Ns,Ms)
    etude = etude_convergence_asiatique(r,T,s0,K,B,Ns,Ms)
    
    print (etude)
    

   
