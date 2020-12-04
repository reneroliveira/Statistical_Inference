import numpy as np
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#np.random.seed(42)



def simulate(theta,n,m):
    #Generate data
    table = np.zeros([n,m+1])
    for i in range(0,n):
        moeda = rd.random()>0.5 #moeda 0 ou 1
        table[i,-1] = moeda
        for j in range(0,m):
            table[i,j] = rd.random()<theta[moeda]
    return table

def B(x,theta_j,k,m):
    #Compute binomial probability
    pm_k = (theta_j[k]**x)*(1-theta_j[k])**(m-x)
    pm_n = (theta_j[(k-1)%2]**x)*(1-theta_j[(k-1)%2])**(m-x)
    return pm_k/(pm_k+pm_n)

def update(theta_j,table,m):
    # Compute a EM iteration
    X = list(table.sum(axis=1))
    Bi = [B(x,theta_j,0,m) for x in X]
    B_prime_i = [B(x,theta_j,1,m) for x in X]
    den1 = m*np.sum(Bi)
    den2 = m*np.sum(B_prime_i)
    num1 = np.sum([xi*bi for xi,bi in zip(X,Bi)])
    num2 = np.sum([xi2*bi2 for xi2,bi2 in zip(X,B_prime_i)])
    theta_1 = num1/den1
    theta_2 = num2/den2
    return [theta_1,theta_2]

def run(theta_0,n,m,p,eps=0.001,max_it=30):
    #Iteractive Process
    table = simulate(p,n,m)
    t = table[:,:-1]
    i = 0
    e = np.inf
    lista = [theta_0]
    while (i<=max_it):
        theta_0= update(theta_0,t,m)
        lista.append(theta_0)
        i+=1
    theta_MLE = MLE(table,m,n)
    return [lista,theta_MLE]

def MLE(table,m,n):
    #Compute Maximum Likelihood Estimator as the proportion of heads
    X_all = table[:,:-1]
    X_head = list(X_all.sum(axis=1))
    M = table[:,-1]
    m1 = n - sum(M)
    m2 = sum(M)
    p1_MLE = sum([X_head[i] for i in range(n) if M[i]==0])/(m1*m)
    p2_MLE = sum([X_head[i] for i in range(n) if M[i]==1])/(m2*m)

    return [p1_MLE,p2_MLE]

    
if __name__=="__main__":
    theta0 = np.random.random([1,2])[0]
    theta0 = [0.4997,0.4999]
    n = 1200
    m = 120
    p = [0.1,0.7]
    max_it = 10
    sim = run(theta0,n,m,p,max_it=max_it)
    
    p1 = [x[0] for x in sim[0]]
    p2 = [x[1] for x in sim[0]]
    p1_MLE = sim[1][0]
    p2_MLE = sim[1][1]
    print("Real p.......")
    print(p)
    print("p_MLE.......")
    print(sim[1])
    print(p1[-1],p2[-2])

    #Generating Figure
    fig,ax = plt.subplots(1,1,figsize=(10,6))
    ax.plot(p1,label="p1_Método_EM",lw=3.5)
    ax.plot(p2,label="p2_Método_EM",lw=3.5)
    ax.plot([p[0]]*(max_it+2),label = "p1_real",ls='--')
    ax.plot([p[1]]*(max_it+2),label = "p2_real",ls='--')

    ax.set_xlabel("j",fontsize=14)
    ax.legend(loc='best',fontsize=14)
    #fig.savefig("sim2-real.jpg")
    #plt.show()

    fig2,ax = plt.subplots(1,1,figsize=(10,6))
    ax.plot(p1,label="p1 estimado",lw=3.5)
    ax.plot(p2,label="p2 estimado",lw=3.5)
    ax.plot([p1_MLE]*(max_it+2),label = "p1_MLE",ls='--')
    ax.plot([p2_MLE]*(max_it+2),label = "p2_MLE",ls='--')

    ax.set_xlabel("j",fontsize=14)
    ax.legend(loc='best',fontsize=14)
    #fig2.savefig("sim2-MLE.jpg")
    #plt.show()


