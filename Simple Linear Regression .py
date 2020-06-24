#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Importing Needed packages
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


# In[7]:


#importing data
df=pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()


# In[8]:


# summarize the data
df.describe()


# In[14]:


#Lets select some features 
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS','FUELCONSUMPTION_HWY']]
cdf.head(9)


# In[15]:


viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB','FUELCONSUMPTION_HWY']]
viz.hist()   #draw feature histograms (for groups of samples)
plt.show()


# In[35]:


#plot CYLINDER vs the Emission, to see how linear is their relation:
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("Emission")
plt.show()


# In[50]:


#Creating train and test dataset
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[51]:


#Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[52]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients and Intercept from(y=mx+c)
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[55]:


#Plot outputs
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r') #x,y=m(coef)*x + c(intercept)
plt.xlabel("Engine size")
plt.ylabel("Emission")


# In[56]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )


# In[ ]:




