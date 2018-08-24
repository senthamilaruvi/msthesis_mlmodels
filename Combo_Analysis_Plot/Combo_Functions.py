
# coding: utf-8

# In[40]:

get_ipython().magic(u'matplotlib inline')


# In[1]:

import numpy as np
import os
import matplotlib.pyplot as plt
import combo
import pandas as pd


# In[3]:

def Predict(Training_data,Test_data):
    
    Data1 = Training_data.values
    Data_test1 = Test_data.values
    
    stdX  = np.std( Data1, 0 )
    index = np.where(stdX !=0)
    meanX = np.mean(Data1[:,index[0]],0)
    
    Data1 = combo.misc.centering(Data1)
    Data_test1 = combo.misc.centering(Data_test1)
    
    X = Data1[:,0:4]
    t = (Data1[:,4])
    
    cov = combo.gp.cov.gauss(X.shape[1], ard = False)
    mean = combo.gp.mean.zero()
    lik = combo.gp.lik.gauss()
    
    gp = combo.gp.model(lik = lik,mean = mean, cov = cov)
    config = combo.misc.set_config()
    np.random.seed(1000)

    index = np.random.permutation(xrange(X.shape[0]))
    train_X = X[index,:]
    train_t = t[index]

    gp.fit(train_X, train_t, config)
    gp.prepare(train_X, train_t)
    
    new_X = Data_test1[:,0:4]
    new_fmean =  gp.get_post_fmean(train_X, new_X)
    new_fcov = gp.get_post_fcov(train_X,new_X)
    sigma = []
    
    for i in range(new_X.shape[0]):
        sigma.append(np.sqrt(new_fcov[i]))
    Predicted_Value = []
    Predicted_Sdev = []
    
    for i in range(0,len(new_fmean)):
        Predicted_Value.append((new_fmean[i]*stdX[4])+meanX[4])
        Predicted_Sdev.append((sigma[i]*stdX[4]))

    Output = pd.DataFrame({'Predicted_Value':Predicted_Value, 'Predicted_Sdev':Predicted_Sdev},
                          columns = ['Predicted_Value', 'Predicted_Sdev'])

    Result = pd.concat([Test_data.reset_index(), Output],axis = 1)

    return(Result)


# In[14]:

def Validate(Data_train,Data_test):
    X = Data_train[:,0:-1]
    t = Data_train[:,-1]
    
    cov = combo.gp.cov.gauss(X.shape[1], ard = False)
    mean = combo.gp.mean.zero()
    lik = combo.gp.lik.gauss()
    
    gp = combo.gp.model(lik = lik,mean = mean, cov = cov)
    
    config = combo.misc.set_config()
    np.random.seed(1000)
    
    index = np.random.permutation(xrange(X.shape[0]))
    train_X = X[index,:]
    train_t = t[index]
    gp.fit(train_X, train_t, config)
    gp.prepare(train_X, train_t)
    
    new_X = Data_test[:,0:-1]
    new_fmean =  gp.get_post_fmean(train_X, new_X)
    
    return (new_fmean)


# In[7]:




# In[ ]:



