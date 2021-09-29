#
# Program pour simuler un Ornstein Uhlenbeck
#

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt 

def browien(t,n):
    '''
        Fonction qui  simule un mouvement brownien entre 
        0 et t avec n pas de discretisation
        W_0 = 0
    '''
    h = float(t)/float(n)
    v = np.sqrt(h)
    normal = np.random.normal(loc=0, scale = v, size=n+1)
    normal[0] = 0

    return np.cumsum(normal)

def ornstein_uhlenbeck(t,n,l,u, x):
    '''
        Simule un processus ornstein_hhlebeck 
        dX_t = Xt*(1-l*(X>0)-m*(X<0))
        avec X_o=x
    '''
    h = float(t)/float(n)
    v = np.sqrt(h)
    normal = np.random.normal(loc=0, scale = v, size=n)
    r = np.zeros(shape = n+1)
    r[0] = x
    for i in np.arange(1,n+1):
        p = r[i-1]
        r[i] = p*(1. - h*(l*(p>0) + u*(p<0))) + normal[i-1]
    
    return r

def barriere(x,b):
    '''
        Calcul (max(abs(x) <= b)
    '''
    m = np.max(np.abs(x))

    return float((m<=b))

def simulation(t,n,l,u,m):
    '''
        simulation de montecarlo pour un instant
        t avec n discretisation, 
        l = lambda
        u = mu
        m = nombre de paths
    '''
    s = 0.
    ss = 0. 

    print ("simulation pour u=%s"%(u))

    for i in np.arange(m):
        x = ornstein_uhlenbeck(t,n,l,u, 0.)
        h = barriere(x,1.)
        s += h
    y = s/float(m)
    d = np.sqrt(y-y*y)
    e = d/np.sqrt(m)
    p = -np.log(y)

    print ('s=%s, p=%s'%(s,p))
    return (p,-np.log(y+1.96*e), -np.log(y-1.96*e))

def simul_normal(t,n,m):
    '''
        Fonction pour simuler toutes les trajectoires d'une normal(0,h)
        d'une simulation de montecarlo de taille m
    '''
    h = float(t)/float(n)
    v = np.sqrt(h)
    normals = np.random.normal(loc=0, scale = v, size=(m,n))

    return  normals

def simul_ornstein_uhlenbeck(t,n,l,u, x,normals):

    '''
        Simule des processus ornstein_hhlebeck 
        dX_t = Xt*(1-l*(X>0)-m*(X<0))
        avec X_o=x
    '''
    h = float(t)/float(n)
    v = np.sqrt(h)
    r = np.zeros(shape = (m,n+1))
    r[:,0] = x
    for i in np.arange(1,n+1):
        p = r[:,i-1]
        r[:,i] = p*(1. - h*(l*(p>0) + u*(p<0))) + normals[:,i-1]
    
    return r

def graphique_processus_ornstein_uhlenbeck(t,n,l,us,x,normals):
    '''
        Créer des trajectoires d'un processus Ornstein uhlenbeck
        asymmétrique
    '''
    ts = np.linspace(start = 0,stop =t,num =n+1)
    for u in us:
        xt = simul_ornstein_uhlenbeck(t,n,l,u, x,normals)
        print ('xt.shape = %s,%s'%xt.shape)
        print('normals.shape = %s,%s'%normals.shape)
        fig,ax = plt.subplots()
        for j in range(normals.shape[0]):
            ax.plot(ts,xt[j,:])
        ax.grid()
        ax.set_title('Simulation OU avec %s'%u)
        ax.set_xlabel('temps')
        fig.savefig('ornstein_uhlenbeck_%d.png'%u)
    
    return 0 

def simul_proba(x):
    '''
        Calcul la probabilité (|x|<1) sur toutes les trajectoires
    '''
    b = np.max(np.abs(x),axis=1)<=1
    m = np.mean(b)
    v = np.std(b)
    p= -np.log(m)
    p1 = -np.log(m+v/np.sqrt(x.shape[0]))
    p2 = -np.log(m-v/np.sqrt(x.shape[0]))

    return (p,p1,p2)


def graphique_running_mean_var(t,n,l,us,x,normals):
    '''
       Calcul de la moyenne et la variance  
    '''
    plt.rcParams["font.size"]=7
    ts = np.linspace(start = 0,stop =t,num =n+1)
    fig,axs = plt.subplots(nrows=us.shape[0], ncols = 3)
    fig.tight_layout()
    print('axs shape {}'.format(axs.shape))
    for u,ax in zip(us,axs):
        xt = simul_ornstein_uhlenbeck(t,n,l,u, x,normals)
        xt_mean = np.mean(xt, axis=0)
        xt_std = np.std(xt,axis=0)
        xt_skew = st.skew(xt,axis=0)
        
        ax[0].plot(ts,xt_mean, label = "Moy u:%d"%u)
        ax[0].grid()
        ax[0].set_xlabel('temps')
        ax[0].set_ylim([-1,1])
        ax[0].set_xlim([0,5])
        ax[0].legend()

        ax[1].plot(ts,xt_std, color='red', label = "Ecart Type u:%d"%u)
        ax[1].grid()
        ax[1].set_xlabel('temps')
        ax[1].set_ylim([0,1.5])
        ax[1].set_xlim([0,5])
        ax[1].legend()

        ax[2].plot(ts,xt_skew, color='blue', label = "Asymétrie u:%d"%u)
        ax[2].grid()
        ax[2].set_xlabel('temps')
        ax[2].set_ylim([-1.5,1.5])
        ax[2].set_xlim([0,5])
        ax[2].legend()


    fig.savefig('ou_analyse_moy_ecart_%d.png'%u)
    
    return 0 





def plot_fig(u,p):
    '''
        graphique de la solution
    '''
    fig,ax = plt.subplots()
    ax.plot(u,p[:,0])
    ax.plot(u,p[:,1], color='red')
    ax.plot(u,p[:,2], color = 'red')
    ax.grid()
    ax.set_xlabel('mu')
    ax.set_title('Simulation (-log(Prob))') 
    fig.show()
    fig.savefig('exo2.png')
    
    return 0

if __name__ == '__main__':

    t=5.
    n=50
    l=1.
    m=100000
    x=0.
    us = np.linspace(start=0,stop=3,num=4, endpoint = True) 
#    ps = [simulation(t,n,l,u,m) for u in us]


    normals = simul_normal(t,n,m)
#    graphique_processus_ornstein_uhlenbeck(t,n,l,us,x,normals)
    us = np.array([0,1,4,10])
    graphique_running_mean_var(t,n,l,us,x,normals)
    ps = []
    for u in us:
        xt = simul_ornstein_uhlenbeck(t,n,l,u, x,normals)
        p = simul_proba(xt)
        print("u=%s, p=%s"%(u,p))
        ps.append(p)

    ps = np.array(ps)
#    plot_fig(us,ps)

