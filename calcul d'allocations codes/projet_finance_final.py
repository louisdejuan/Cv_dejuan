# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:42:50 2021

@author: Louis DE JUAN
"""

import numpy as np
import matplotlib.pyplot as plt



def brownien_taux_instantane(mu,vol,temps):
    '''
        Fonction qui renvoie : dr = u dt + v * sqrt(delta_t) * N(0,1)
        params : 
            mu : taux de rendement (annuelle)
            v  : volatilité (annuelle)
            delta_t : periode (annuelle)
            taille : taille de la trajectoire à simuler
        returns :   
            le rendement instantane
    '''
    taille = temps.shape[0]
    delta_t = np.diff(temps, prepend = 0.)

    result = mu * delta_t + vol * np.sqrt(delta_t) * np.random.normal(0.,1., taille)
    result [0] = 0. 
    
    return result 

def calcul_contribution(income, r_income, r_contrib, vect_temps) : 
    '''
        Fonction calcul la contribution annuelle à chaque début d'annees
        params: 
            income : salaire initial
            r_income : rendement annuel du salaire
            r_contrib : taux de contribution
            vect_temps : dates
    '''
    result = np.full(shape = vect_temps.shape, fill_value = 0.)
    bond = 1.
    result[0] = income * r_contrib * bond
    for i in np.arange(1,result.shape[0]):
        if (vect_temps[i] == np.int(vect_temps[i])):
            #pour chaque annee
            
            bond = bond * (1.+r_income)
            result[i] = income * bond * r_contrib
    
    return result
   
def calcul_allocation_constante(r_highs,r_lows, a_high, a_low, contributions, npv):
    '''
        Calcul de la valeur de l'investisseme pour une allocation constante
        params : 
            r_highs : vecteur de rendement de l'actif risqué
            r_lows  : vecteur de rendement de l'actif non risqué
            a_high : allocatoin sur l'actif risqué
            a_low : allocatoin sur l'actif non risqué
            contributions : contributions sur chaque période
            npv : valeur du portefeuille initial (contribution initiale)
    '''
    r_port = a_high * r_highs[0] + a_low * r_lows[0]
    npv_next = npv * (1 + r_port) + contributions[0]
    for i in np.arange(1, r_highs.shape[0]):
        #les coeffs sont constant 
        r_port = a_high * r_highs[i] + a_low * r_lows[0]        
        npv_next = npv * (1.+r_port) + contributions[i]
        npv = npv_next
    return npv

def calcul_allocation_temporelle(r_highs,r_lows, a_high, a_low, temps, contributions, npv):
    '''
        Calcul de la valeur de l'investisseme pour une allocation en fonction du temps
        params : 
            r_highs : vecteur de rendement de l'actif risqué
            r_lows  : vecteur de rendement de l'actif non risqué
            a_high : allocatoin sur l'actif risqué
            a_low : allocatoin sur l'actif non risqué
            temps : liste des années
            contributions : contributions sur chaque période
            npv : valeur du portefeuille initial (contribution initiale)
    '''
    r_port = a_high * r_highs[0] + a_low * r_lows[0]
    npv_next = npv * (1 + r_port) + contributions[0]
    for i in np.arange(1, r_highs.shape[0]):
        r_port = a_high * r_highs[i] + a_low * r_lows[0]        
        npv_next = npv * (1.+r_port) + contributions[i]
        npv = npv_next
#
#     Mise a jour des coefficients
#       
        if (temps[i] == np.int(temps[i])): #permet juste de prendre le premier jour des années
            if (temps[i]<20):
                a_high=0.8
                a_low=0.2
            if(20<=temps[i]<30):
                a_high=0.5
                a_low=0.5
            if(temps[i]>=30):
                a_high=0.15
                a_low=0.85
                
                
    return npv
    
def calcul_allocation_cppi(r_highs,r_lows, m_high, pr_floor, contributions, temps, npv):
    '''
        Calcul de la valeur de l'investissement pour une allocation cppi
        params : 
            r_highs : vecteur de rendement de l'actif risqué
            r_lows  : vecteur de rendement de l'actif non risqué
            m : multiplicateur d'allocation sur l'actif risque
            pr_floor : niveau du floor lorsque : il sera mis à jour à chaque année
            temps : vecteur de temps
            contributions : contributions sur chaque période
            npv : valeur du portefeuille initial (contribution initiale)
    '''
#
# Calcul du coussin et des allocations actif risqué et actif sans risque
#   
    npv_next = npv + contributions[0]
    npv = npv_next
    floor = npv * pr_floor
    coussin = npv - floor
    a_high = min(max(0,coussin / npv * m_high),1)
    a_low = 1. - a_high
    
    for i in np.arange(1, r_highs.shape[0]):
        r_port = a_high * r_highs[i] + a_low * r_lows[0]        
        npv_next = npv * (1.+r_port) + contributions[i]
        npv = npv_next
#
# Mise a jour du floor (uniquement annuellement)
#       
        if (temps[i] == np.int(temps[i])): 
            floor_new = npv * pr_floor
            if (floor_new > floor) : 
                floor = floor_new
#
# Mise a jour des nouvelles allocations : il faut les calculer après de calculer 
#le rendement du portefeuille pour la date suivante
# Sinon pas causal 
#       
        coussin = npv - floor
        a_high = min(max(0,coussin / npv * m_high),1)
        a_low = 1. - a_high
            
    
    return npv

def histogramme(npvs, titre) :

    
    fig,ax = plt.subplots()
    ax.hist(npvs, 50, density=True, facecolor='g', alpha=0.75)
    ax.grid(True)
    ax.set_title(titre)
    fig.show()
    fig.savefig('%s.png'%titre)
    
    return 0
    
if __name__ == '__main__':
    
    P=0; #portefeuille initial
    periode = 12 # Douze mois
    nombre_mois = 40 * periode # nombre d'annees de l'investissement pour la retraite
     
    temps=np.linspace(0.,40., np.int(nombre_mois+1))
# calcul des valeurs fixes: 
    contributions = calcul_contribution(50000., 0.03, 0.1, temps)
    print(contributions)
    
# Calcul des stratégies
#   Strategies 1 : Allocation constante
#   Strategies 2 : Allocation CPPI 
    
    npv_alloc_constantes = []
    npv_alloc_cppis = []
    npv_alloc_tempo=[]
    
    for i in np.arange(10000):
        
            
        r_highrisks = brownien_taux_instantane(0.06,0.2, temps)
        r_lowrisks = brownien_taux_instantane(0.02,0.05, temps)
        
      
        if not(i%1000):
            print ('Simmulation %s'%i)
            
        
        npv_alloc_constantes.append(calcul_allocation_constante(r_highrisks,r_lowrisks, 0.4, 0.6, contributions, 0.))
        npv_alloc_cppis.append(calcul_allocation_cppi(r_highrisks,r_lowrisks, 5.0, 0.8, contributions, temps, 0.))
        npv_alloc_tempo.append(calcul_allocation_temporelle(r_highrisks,r_lowrisks, 0.8, 0.2, temps, contributions, 0.))           
    
    print ('Moyenne : Allocation Constante %s, Allocation CPPI %s, Allocation rebalancing %s'%(np.average(npv_alloc_constantes), np.average(npv_alloc_cppis),np.average(npv_alloc_tempo))) 
    print ('Ecart_type : Allocation Constante %s, Allocation CPPI %s, Allocation rebalancing %s'%(np.std(npv_alloc_constantes), np.std(npv_alloc_cppis), np.std(npv_alloc_tempo))) 
    print ('Percentile_0.05 : Allocation Constante %s, Allocation CPPI %s, Allocation rebalancing %s'%(np.percentile(npv_alloc_constantes,5), np.percentile(npv_alloc_cppis,5), np.percentile(npv_alloc_tempo,5)))
    print ('Percentile_0.95 : Allocation Constante %s, Allocation CPPI %s, Allocation rebalancing %s'%(np.percentile(npv_alloc_constantes,95), np.percentile(npv_alloc_cppis,95), np.percentile(npv_alloc_tempo,95)))

    histogramme(npv_alloc_constantes, 'allocation_constante')
    histogramme(npv_alloc_cppis, 'allocation_cppi')
    histogramme(npv_alloc_tempo,'allocation_rebalancing')
    
    
    
     
    
    




    

    
    