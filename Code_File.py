
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_selection import chi2, SelectKBest


# In[2]:


# Path to the data file
file_path = r"D:\Data_Sets\Mobile_Prices\data.csv"

# Reading the data from the Southern Second Order file, and also passing the column names to south_data data frame
south_data = pd.read_csv(file_path)


# Printing the number of data points and the number of columns of south_data data frame
print("The number of data points in the data  :", south_data.shape[0])
print("The features of the data :", south_data.shape[1])

# Printing the head of south_data data frame
south_data.head(5)


# In[3]:


south_data.isnull().sum()


# In[5]:


x = south_data.drop("tss", axis = 1)
y = south_data["tss"]

bestfit = SelectKBest(score_func=chi2, k=5)
features = bestfit.fit(x,y)
x_new = features.transform(x)

features.scores_

