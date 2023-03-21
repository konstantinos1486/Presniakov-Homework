#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import io
import statsmodels.formula.api as smf


# In[4]:


data = '''sleep totwrk age male hrwage
3113   3438  32   1    7.07 
2920   5020  31   1    1.43 
2670   2815  44   1   20.53 
3083   3786  30   0    9.62 
3448   2580  64   1    2.75 
4063   1205  41   1   19.25 '''


# In[5]:


data


# In[6]:


df = pd.read_csv(io.StringIO(data), sep='\s+')
df


# In[7]:


df ['pred'] = 3525.14  -0.16*df['totwrk']+1.62*df['age']+51.84*df['male']-9.88*np.log(df['hrwage'])
df


# In[8]:


df['residual']=df['sleep']-df['pred']
df


# In[ ]:




