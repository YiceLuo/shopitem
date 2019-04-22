#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn import datasets, linear_model
import hdbscan
import seaborn as sns
import statsmodels.api as sm
from numpy import linalg
from math import sqrt
import time
from sklearn.decomposition import PCA
from skopt import gp_minimize, dump, load
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
from skopt.utils import use_named_args
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv('shopitem.csv')
Shopgroup = pd.read_csv('Shopgroup.csv')
Itemgroup = pd.read_csv('Itemgroup.csv')


# In[3]:


sparsity=1-dataset.shape[0]/214200/60
sparsity


# In[4]:


Ntr=int(dataset.shape[0]*0.6)
Nv=int(dataset.shape[0]*0.8)
sim_data=dataset.values
n=214200
m=60
train=sim_data[:Ntr,:]
validation=sim_data[Ntr:Nv,:]
test=sim_data[Nv:,:]
x_train=train[:,:2]  
x_test=test[:,:2]
x_valid=validation[:,:2]
x_train=x_train.astype(int)
x_test=x_test.astype(int)
x_valid=x_valid.astype(int)
y=train[:,2]
y_test=test[:,2]
y_valid=validation[:,2]


# In[5]:


#player set
ind1=[] #matched played for each user
y1=[]
#match set
ind2=[] #players in this match
y2=[]
for u in range(n):
    ind1.append(x_train[x_train[:,0]==u,1])
    y1.append(train[x_train[:,0]==u,2])
for i in range(m):
    ind2.append(x_train[x_train[:,1]==i,0])
    y2.append(train[x_train[:,1]==i,2])


# In[6]:


def shrink(x,l):
    if x>l/2:
        X=x-l/2
    elif x<-l/2:
        X=x+l/2
    else:
        X=0
    return X
Vshrink= np.vectorize(shrink)


# In[7]:


def mylasso(y,x,k,l,L):
    betaols=linalg.solve(x.transpose()@x+L*np.identity(k),x.transpose()@y)
    beta = Vshrink(betaols,l)
    return beta


# In[8]:


def myl0(y,x,k,l,L):
    beta=linalg.solve(x.transpose()@x+L*np.identity(k),x.transpose()@y)
    beta = beta*((beta>np.median(beta))+1)
    return beta


# In[ ]:


def mygsm(foo,ind1,y1,ind2,y2,x_train,x_valid,y,y_valid,matchgroup,playergroup,n,m,k,l,L): 
    inM=matchgroup.In.max()
    exM=matchgroup.Ex.max()
    inN=playergroup.In.max()
    exN=playergroup.Ex.max()
    P=np.random.normal(2,1,(n,k))
    Q=np.random.normal(2,1,(m,k))
    S=np.random.normal(2,1,(n,k))
    T=np.random.normal(2,1,(m,k))
    A=np.random.normal(2,1,(n,k))
    B=np.random.normal(2,1,(m,k))
    Pnew=np.zeros(shape=(n,k))
    Qnew=np.zeros(shape=(m,k))
    Snew=np.zeros(shape=(n,k))
    Tnew=np.zeros(shape=(m,k))
    Anew=np.zeros(shape=(n,k))
    Bnew=np.zeros(shape=(m,k))
    yhat=np.sum(np.multiply((P[x_train[:,0],:]+S[x_train[:,0],:]+A[x_train[:,0],:]),(Q[x_train[:,1],:]+T[x_train[:,1],:]+B[x_train[:,1],:])),1)
    it=1  #number of iterations
    diff=1  #improvement over last iteration
    diffQTB=1
    diffPSA=1
    while(diff>1e-5 or it<10):
        print("diff=",diff)
        diff=np.sum(np.multiply(Pnew+Snew+Anew-P-S-A,Pnew+Snew+Anew-P-S-A))/n/k+np.sum(np.multiply(Qnew+Tnew+Bnew-Q-T-B,Qnew+Tnew+Bnew-Q-T-B))/m/k
        while(diffQTB>1e-8):
            for i in range(m):
                xpsa=P[ind2[i],:]+S[ind2[i],:]+A[ind2[i],:]
                r=y2[i]-(P[ind2[i],:]+S[ind2[i],:]+A[ind2[i],:])@(T[i,:]+B[i,:])
                Qnew[i,:]=foo(r,xpsa,k,l,L)

            for i in range(inM):
                I=matchgroup.matchId[matchgroup.In==i]
                I=I[I<len(ind2)]
                playerI=[]
                for j in I:
                    playerI=np.concatenate([playerI,ind2[j]]).astype("int")
                xpsa=P[playerI,:]+S[playerI,:]+A[playerI,:]
                r=[]
                for j in I:
                    R=y2[j]-(P[ind2[j],:]+S[ind2[j],:]+A[ind2[j],:])@(Qnew[j,:]+B[j,:])
                    r=np.concatenate([r,R])
                Tnew[I,:]=foo(r,xpsa,k,l,L)

            for i in range(exM):
                I=matchgroup.matchId[matchgroup.Ex==i]
                I=I[I<len(ind2)]
                playerI=[]
                for j in I:
                    playerI=np.concatenate([playerI,ind2[j]]).astype("int")
                xpsa=P[playerI,:]+S[playerI,:]+A[playerI,:]
                r=[]
                for j in I:
                    R=y2[j]-(P[ind2[j],:]+S[ind2[j],:]+A[ind2[j],:])@(Qnew[j,:]+Tnew[j,:])
                    r=np.concatenate([r,R])
                Bnew[I,:]=foo(r,xpsa,k,l,L)
           
            diffQTB=np.sum(np.multiply(Qnew-Q,Qnew-Q))/m/k+np.sum(np.multiply(Tnew-T,Tnew-T))/inM/k+np.sum(np.multiply(Bnew-B,Bnew-B))/exM/k
            print("diffQTB=",diffQTB)
            Q=Qnew
            T=Tnew
            B=Bnew
        while(diffPSA>1e-8):
            for i in range(n):
                xqtb=Q[ind1[i],:]+T[ind1[i],:]+B[ind1[i],:]
                r=y1[i]-(Q[ind1[i],:]+T[ind1[i],:]+B[ind1[i],:])@(S[i,:]+A[i,:])
                Pnew[i,:]=foo(r,xqtb,k,l,L)

            for i in range(inN):
                I=playergroup.playerId[playergroup.In==i]
                I=I[I<len(ind1)]
                matchI=[]
                for j in I:
                    matchI=np.concatenate([matchI,ind1[j]]).astype("int")
                xqtb=Q[matchI,:]+T[matchI,:]+B[matchI,:]
                r=[]
                for j in I:
                    R=y1[j]-(Q[ind1[j],:]+T[ind1[j],:]+B[ind1[j],:])@(Pnew[j,:]+A[j,:])
                    r=np.concatenate([r,R])
                Snew[I,:]=foo(r,xqtb,k,l,L)

            for i in range(exN):
                I=playergroup.playerId[playergroup.Ex==i]
                I=I[I<len(ind1)]
                matchI=[]
                for j in I:
                    matchI=np.concatenate([matchI,ind1[j]]).astype("int")
                xqtb=Q[matchI,:]+T[matchI,:]+B[matchI,:]
                r=[]
                for j in I:
                    R=y1[j]-(Q[ind1[j],:]+T[ind1[j],:]+B[ind1[j],:])@(Pnew[j,:]+Snew[j,:])
                    r=np.concatenate([r,R])
                Anew[I,:]=foo(r,xqtb,k,l,L)  
            diffPSA=np.sum(np.multiply(Pnew-P,Pnew-P))/n/k+np.sum(np.multiply(Snew-S,Snew-S))/inN/k+np.sum(np.multiply(Anew-A,Anew-A))/exN/k
            print("diffPSA=",diffPSA)
            P=Pnew
            S=Snew
            A=Anew
        it=it+1
    
    yhat_valid=np.sum(np.multiply((P[x_valid[:,0],:]+S[x_valid[:,0],:]+A[x_valid[:,0],:]),(Q[x_valid[:,1],:]+T[x_valid[:,1],:]+B[x_valid[:,1],:])),1)
    #yhat_valid=np.round(yhat_valid,decimals=4)
    RMSE=sqrt((y_valid-yhat_valid)@(y_valid-yhat_valid)/y_valid.size)
    return RMSE,yhat_valid


# In[ ]:


def mygsm1(foo,ind1,y1,ind2,y2,x_train,x_valid,y,y_valid,matchgroup,playergroup,n,m,k,l,L): 
    inM=matchgroup.In.max()
    #exM=matchgroup.Ex.max()
    inN=playergroup.In.max()
    #exN=playergroup.Ex.max()
    P=np.random.normal(2,1,(n,k))
    Q=np.random.normal(2,1,(m,k))
    S=np.random.normal(2,1,(n,k))
    T=np.random.normal(2,1,(m,k))
    #A=np.random.normal(2,1,(n,k))
    #B=np.random.normal(2,1,(m,k))
    Pnew=np.zeros(shape=(n,k))
    Qnew=np.zeros(shape=(m,k))
    Snew=np.zeros(shape=(n,k))
    Tnew=np.zeros(shape=(m,k))
    #Anew=np.zeros(shape=(n,k))
    #Bnew=np.zeros(shape=(m,k))
    yhat=np.sum(np.multiply((P[x_train[:,0],:]+S[x_train[:,0],:]),(Q[x_train[:,1],:]+T[x_train[:,1],:])),1)
    it=1  #number of iterations
    diff=1  #improvement over last iteration
    diffQTB=1
    diffPSA=1
    while(diff>1e-5 or it<10):
        print("diff=",diff)
        diff=np.sum(np.multiply(Pnew+Snew-P-S,Pnew+Snew-P-S))/n/k+np.sum(np.multiply(Qnew+Tnew-Q-T,Qnew+Tnew-Q-T))/m/k
        while(diffQTB>1e-8):
            for i in range(m):
                xpsa=P[ind2[i],:]+S[ind2[i],:]
                r=y2[i]-(P[ind2[i],:]+S[ind2[i],:])@(T[i,:])
                Qnew[i,:]=foo(r,xpsa,k,l,L)

            for i in range(inM):
                I=matchgroup.matchId[matchgroup.In==i]
                I=I[I<len(ind2)]
                playerI=[]
                for j in I:
                    playerI=np.concatenate([playerI,ind2[j]]).astype("int")
                xpsa=P[playerI,:]+S[playerI,:]
                r=[]
                for j in I:
                    R=y2[j]-(P[ind2[j],:]+S[ind2[j],:])@(Qnew[j,:])
                    r=np.concatenate([r,R])
                Tnew[I,:]=foo(r,xpsa,k,l,L)

           
            diffQTB=np.sum(np.multiply(Qnew-Q,Qnew-Q))/m/k+np.sum(np.multiply(Tnew-T,Tnew-T))/inM/k
            print("diffQTB=",diffQTB)
            Q=Qnew
            T=Tnew
        while(diffPSA>1e-8):
            for i in range(n):
                xqtb=Q[ind1[i],:]+T[ind1[i],:]
                r=y1[i]-(Q[ind1[i],:]+T[ind1[i],:])@(S[i,:])
                Pnew[i,:]=foo(r,xqtb,k,l,L)

            for i in range(inN):
                I=playergroup.playerId[playergroup.In==i]
                I=I[I<len(ind1)]
                matchI=[]
                for j in I:
                    matchI=np.concatenate([matchI,ind1[j]]).astype("int")
                xqtb=Q[matchI,:]+T[matchI,:]
                r=[]
                for j in I:
                    R=y1[j]-(Q[ind1[j],:]+T[ind1[j],:])@(Pnew[j,:])
                    r=np.concatenate([r,R])
                Snew[I,:]=foo(r,xqtb,k,l,L)

            diffPSA=np.sum(np.multiply(Pnew-P,Pnew-P))/n/k+np.sum(np.multiply(Snew-S,Snew-S))/inN/k
            print("diffPSA=",diffPSA)
            P=Pnew
            S=Snew
        it=it+1
    
    yhat_valid=np.sum(np.multiply((P[x_valid[:,0],:]+S[x_valid[:,0],:]),(Q[x_valid[:,1],:]+T[x_valid[:,1],:])),1)
    #yhat_valid=np.round(yhat_valid,decimals=4)
    RMSE=sqrt((y_valid-yhat_valid)@(y_valid-yhat_valid)/y_valid.size)
    #MAE=sum(abs((y_valid-yhat_valid)))/y_valid.size
    return RMSE,yhat_valid


# In[ ]:


def mygsm0(foo,ind1,y1,ind2,y2,x_train,x_valid,y,y_valid,n,m,k,l,L): 
    P=np.random.normal(2,1,(n,k))
    Q=np.random.normal(2,1,(m,k))
    Pnew=np.zeros(shape=(n,k))
    Qnew=np.zeros(shape=(m,k))
    yhat=np.sum(np.multiply((P[x_train[:,0],:]),(Q[x_train[:,1],:])),1)
    it=1  #number of iterations
    diff=1  #improvement over last iteration
    diffQTB=1
    diffPSA=1
    while(diff>1e-5 or it<10):
        print("diff=",diff)
        diff=np.sum(np.multiply(Pnew-P,Pnew-P))/n/k+np.sum(np.multiply(Qnew-Q,Qnew-Q))/m/k
        while(diffQTB>1e-8):
            for i in range(m):
                xpsa=P[ind2[i],:]
                r=y2[i]
                Qnew[i,:]=foo(r,xpsa,k,l,L)   
            diffQTB=np.sum(np.multiply(Qnew-Q,Qnew-Q))/m/k
            print("diffQTB=",diffQTB)
            Q=Qnew
        while(diffPSA>1e-8):
            for i in range(n):
                xqtb=Q[ind1[i],:]
                r=y1[i]
                Pnew[i,:]=foo(r,xqtb,k,l,L)

            diffPSA=np.sum(np.multiply(Pnew-P,Pnew-P))/n/k+np.sum(np.multiply(Snew-S,Snew-S))/inN/k
            print("diffPSA=",diffPSA)
            P=Pnew
            S=Snew
        it=it+1
    
    yhat_valid=np.sum(np.multiply((P[x_valid[:,0],:]),(Q[x_valid[:,1],:])),1)
    #yhat_valid=np.round(yhat_valid,decimals=4)
    RMSE=sqrt((y_valid-yhat_valid)@(y_valid-yhat_valid)/y_valid.size)
    #MAE=sum(abs((y_valid-yhat_valid)))/y_valid.size
    return RMSE,yhat_valid


# 2-layered gssvd rmse focused k =4,...,10

# In[ ]:


l = Real(low=1e-7, high=1, prior='uniform',
                             name='l')
L = Real(low=1e-7, high=1, prior='uniform',
                             name='L')

dimensions = [l,L]


# In[ ]:


@use_named_args(dimensions=dimensions)
def Fitness(l,L):
    print()
    rmse,yhat = mygsm(mylasso,ind1,y1,ind2,y2,x_train,x_valid,y,y_valid,Shopgroup,Itemgroup,
                      n,m,k,l,L)
    print()
    print("rmse:",rmse)
    print()
    return rmse


# In[ ]:


start=time.time()
rmse=np.zeros(shape=(3,7))
for k in range(4,11):
    default_parameters = [1e-2,1e-2]
    search_result = gp_minimize(func=Fitness,
                            dimensions=dimensions,
                            acq_func='EI', 
                            n_calls=100,
                            x0=default_parameters)
    para=search_result.x
    rmse[0,k-4]=mygsm(mylasso,ind1,y1,ind2,y2,x_train,x_test,y,y_test,Shopgroup,Itemgroup,n,m,k,para[0],
    para[1])[0]
print(time.time()-start)


# 1-layered gssvd rmse focused 

# In[ ]:


@use_named_args(dimensions=dimensions)
def Fitness(l,L):
    rmse,yhat = mygsm1(mylasso,ind1,y1,ind2,y2,x_train,x_valid,y,y_valid,Shopgroup,Itemgroup,
                      n,m,k,l,L)
    print()
    print("rmse:",rmse)
    print()
    return rmse


# In[ ]:


start=time.time()
for k in range(4,11):
    default_parameters = [1e-2,1e-2]
    search_result = gp_minimize(func=Fitness,
                            dimensions=dimensions,
                            acq_func='EI', 
                            n_calls=100,
                            x0=default_parameters)
    para=search_result.x
    rmse[1,k-4]=mygsm1(mylasso,ind1,y1,ind2,y2,x_train,x_test,y,y_test,Shopgroup,Itemgroup,n,m,k,para[0],
    para[1])[0]
print(time.time()-start)


# nogroupsvd rmse focused

# In[ ]:


@use_named_args(dimensions=dimensions)
def Fitness(l,L):
    print()
    rmse,yhat = mygsm0(mylasso,ind1,y1,ind2,y2,x_train,x_valid,y,y_valid,
                      n,m,k,l,L)
    print()
    print("rmse:",rmse)
    print()
    return rmse


# In[ ]:


start=time.time()
for k in range(4,11):
    default_parameters = [1e-2,1e-2]
    search_result = gp_minimize(func=Fitness,
                            dimensions=dimensions,
                            acq_func='EI', 
                            n_calls=100,
                            x0=default_parameters)
    para=search_result.x
    rmse[2,k-4]=mygsm0(mylasso,ind1,y1,ind2,y2,x_train,x_test,y,y_test,n,m,k,para[0],
    para[1])[0]
print(time.time()-start)


# 2-layered gssvd recommender focused

# In[ ]:


@use_named_args(dimensions=dimensions)
def fitness(l,L):
    Y = mygsm(myl0,ind1,y1,ind2,y2,x_train,x_valid,y,y_valid,Shopgroup,Itemgroup,
                      n,m,k,l,L)[1]
    b=y_valid>(np.median(y_valid))
    acc=[]
    for i in range(1,10):
        a=Y>=(Y[np.argsort(Y)[-i]])
        acc=np.append(acc,sum(b[a])/len(b[a]))
    return acc.mean()


# In[ ]:


start=time.time()
accuracy=np.zeros(shape=(3,7))
for k in range(4,11):
    default_parameters = [0.5,0.5]
    search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', 
                            n_calls=100,
                            x0=default_parameters)
    para=search_result.x
    
    a=mygsm(myl0,ind1,y1,ind2,y2,x_train,x_test,y,y_test,Shopgroup,Itemgroup,n,m,k,para[0],
    para[1])
    b=y_valid>(np.median(y_valid))
    acc=[]
    for i in range(1,10):
        a=Y>=(Y[np.argsort(Y)[-i]])
        acc=np.append(acc,sum(b[a])/len(b[a]))
    accuracy[0,k-4]=acc.mean()
print(time.time()-start)


# 1-layered gssvd recommender focused

# In[ ]:


@use_named_args(dimensions=dimensions)
def fitness(l,L):
    Y = mygsm1(myl0,ind1,y1,ind2,y2,x_train,x_valid,y,y_valid,Shopgroup,Itemgroup,
                      n,m,k,l,L)[1]
    b=y_valid>(np.median(y_valid))
    acc=[]
    for i in range(1,10):
        a=Y>=(Y[np.argsort(Y)[-i]])
        acc=np.append(acc,sum(b[a])/len(b[a]))
    return acc.mean()


# In[ ]:


start=time.time()
for k in range(4,11):
    default_parameters = [0.5,0.5]
    search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', 
                            n_calls=100,
                            x0=default_parameters)
    para=search_result.x
    
    a=mygsm1(myl0,ind1,y1,ind2,y2,x_train,x_test,y,y_test,Shopgroup,Itemgroup,n,m,k,para[0],
    para[1])
    b=y_valid>(np.median(y_valid))
    acc=[]
    for i in range(1,10):
        a=Y>=(Y[np.argsort(Y)[-i]])
        acc=np.append(acc,sum(b[a])/len(b[a]))
    accuracy[1,k-4]=acc.mean()
print(time.time()-start)


# svd recommender focused

# In[ ]:


@use_named_args(dimensions=dimensions)
def fitness(l,L):
    Y = mygsm0(myl0,ind1,y1,ind2,y2,x_train,x_valid,y,y_valid,
                      n,m,k,l,L)[1]
    b=y_valid>(np.median(y_valid))
    acc=[]
    for i in range(1,10):
        a=Y>=(Y[np.argsort(Y)[-i]])
        acc=np.append(acc,sum(b[a])/len(b[a]))
    return acc.mean()


# In[ ]:


start=time.time()
for k in range(4,11):
    default_parameters = [0.5,0.5]
    search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', 
                            n_calls=100,
                            x0=default_parameters)
    para=search_result.x
    
    a=mygsm0(myl0,ind1,y1,ind2,y2,x_train,x_test,y,y_test,n,m,k,para[0],
    para[1])
    b=y_valid>(np.median(y_valid))
    acc=[]
    for i in range(1,10):
        a=Y>=(Y[np.argsort(Y)[-i]])
        acc=np.append(acc,sum(b[a])/len(b[a]))
    accuracy[2,k-4]=acc.mean()
print(time.time()-start)


# In[ ]:




