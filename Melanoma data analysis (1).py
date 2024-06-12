#!/usr/bin/env python
# coding: utf-8

# ### Context ###
# 
# The data consists of measurements made on patients with malignant melanoma. Each patient
# had their tumour removed by surgery at the Department of Plastic Surgery, University Hospital
# of Odense, Denmark during the period 1962 to 1977. The surgery consisted of complete
# removal of the tumour together with about 2.5cm of the surrounding skin.
# Among the measurements taken were the thickness of the tumour and whether it was
# ulcerated or not. These are thought to be important prognostic variables in that patients with a
# thick and/or ulcerated tumour have an increased chance of death from melanoma. Patients
# were followed until the end of 1977. The data frame contains the following columns.
# 
# • time - Survival time in days since the operation.
# 
# • status - The patients status at the end of the study
# 
# 1 indicates that they had died from melanoma, 2 indicates that they were still alive and 3
# indicates that they had died from causes unrelated to their melanoma.
# 
# • sex - The patients sex; 1=male, 0=female.
# 
# • age - Age in years at the time of the operation.
# 
# • year - Year of operation.
# 
# • thickness - Tumour thickness in mm.
# 
# • ulcer - Indicator of ulceration; 1=present, 0=absent

# ## Task
# As a data scientist, you are tasked to ask salient questions,analyse the data and report your findings. 
# 
# 
# 
# 

# ### OBJECTIVES / PROBLEM STATEMENT
# 
# * The aim is to analyse the survival rate of malignant melanoma patients after the tumour surgery
# 

# ### Import the required libraries

# In[18]:


#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns


# In[97]:


import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import scipy.stats as stats 
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
#import pingouin
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


ls


# ### Import your data

# In[20]:


df_melanoma = pd.read_csv('melanoma.csv')


# ### Sanity checks

# In[21]:


df_melanoma.head()


# In[22]:


df_melanoma.tail()


# In[23]:


df_melanoma.info()


# In[24]:


df_melanoma.describe()


# In[25]:


df_melanoma.shape


# In[26]:


df_melanoma.dtypes


# In[27]:


for col in df_melanoma.columns :
    print(col)


# In[28]:


df_melanoma.isna().sum()


# ### Data Cleaning

# In[29]:


df_melanoma['status'].replace({3 :'unrelated_cause', 2 :'alive', 1 :'died'},inplace = True)


# In[30]:


df_melanoma['sex'].replace ({1 :'Male', 0 :'Female'},inplace = True)


# In[31]:


df_melanoma['ulcer'].replace({1 :'present', 0:'absent'}, inplace = True)                  


# In[57]:


df_melanoma.head()


# In[39]:


df_melanoma.dtypes


# In[50]:


df_melanoma.nunique()


# ### EXPLORATORY DATA ANALYSIS

# In[51]:


# Total number of people who survived after the surgery
survived = df_melanoma[df_melanoma['status'] == 'alive']
total_survived = survived['status'].count()

print(f'{total_survived} people were alive after the surgery' )


# In[52]:


# Total number of people who died after the surgery
died = df_melanoma[df_melanoma['status'] == 'died']
total_died = died['status'].count()

print(f'{total_died} people died after the surgery' )


# In[53]:


#Total number of people who died from causes unrelated to their melanoma
unrelated_cause = df_melanoma[df_melanoma['status']== 'unrelated_cause']
total_unrelated = unrelated_cause['status'].count()

print(f'{total_unrelated} people died from causes unrelated to their melanoma' )


# In[54]:


# Number of males who survived
number_males = df_melanoma[df_melanoma['sex'] == 'Male']
total_males = number_males['sex'].count()

print(f'{total_males} number of males survived')


# In[55]:


# Number of females who survived
number_females = df_melanoma[df_melanoma['sex'] == 'Female']
total_females = number_females['sex'].count()

print(f'{total_females} number of males survived')


# In[65]:


#Ulceration Indication

ulcer_indicator1 = df_melanoma[df_melanoma['ulcer']=='present']
ulcer_present = ulcer_indicator1['ulcer'].count()

print(ulcer_present, 'number of people who were indicated to have ulcer is present')


# In[66]:


ulcer_indicator2 = df_melanoma[df_melanoma['ulcer']=='absent']
ulcer_absent = ulcer_indicator2['ulcer'].count()

print(ulcer_absent,'The number of people who were indicated to have ulcer is absent')   


# In[67]:


df_melanoma.describe(include = 'all')


# ### Plotting the distributions for numerical variables (Time,Age,Year,Thickness)

# In[37]:


# Set up the figure and axes for subplots
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 6))  # Adjusted to have only one row and three columns

# Numeric Variables
sns.histplot(df_melanoma['time'], ax=axes[0], kde=True, color='violet')
axes[0].set_title('Distribution of Time')

sns.histplot(df_melanoma['age'], ax=axes[1], kde=True, color='skyblue')
axes[1].set_title('Distribution of their Age')

sns.histplot(df_melanoma['year'], ax=axes[2], kde=True, color='green')
axes[2].set_title('Distribution of Year')

sns.histplot(df_melanoma['thickness'],ax =axes[3],kde=True,color = 'pink')
axes[3].set_title('Distribution of Thickness')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()


# **Observation for Time**
# * Time has a normal distribution
# * It can be observed that most people lived up to 2000 days after the operation.
# 
# **Observation for Age**
# * Age has a normal distribution
# * It can be observed that most people between ages 50 and 70 were operated.
# 
# **Observation for Year**
# * Year has a normal distribution
# * In 1972 most people were operated
# 
# **Observation for Thickness**
# * Thickness distribution is skewed to the right
# * The highest thickness was between 0.0mm to 2.5mm

# ### Plotting the distributions for categorical variables (Sex,Status,Ulcer)

# In[38]:


# Set up the figure and axes for subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Categorical Variables
sns.countplot(x='sex', data=df_melanoma, ax=axes[0], palette='pastel')
axes[0].set_title('Number of Females and Males')


status_counts = df_melanoma['status'].value_counts().index

 
sns.countplot(x='status', data=df_melanoma, order=status_counts, ax=axes[1], palette='colorblind')
axes[1].set_title('Orders by Cuisine Type')


sns.countplot(x='ulcer', data=df_melanoma, ax=axes[2], palette='pastel')
axes[2].set_title('Number of Ulceration Patients')

plt.tight_layout()

plt.show()


# **Observation for Sex**
# * Females had the highest number of people who were operated thus over 120
# * Males were the lowest 79
# 
# **Observation for Status**
# * It can be observed that the number of people who were alive was high even though they had melanoma
# * 58 people died from melanoma
# * The lowest was the number of people who from unrelated cause
# 
# **Observation for Ulcer**
# * The number of people who did not have Ulcer was high.

# ### Bivariate/Multivariate Analysis

# In[46]:


df_melanoma.head()


# In[47]:


# Box plot of Delivery Time by Day of the Week
plt.figure(figsize=(10, 6))
sns.boxplot(x='status', y='time', data=df_melanoma, palette='pastel')
plt.title('Box Plot of time by status')
plt.xlabel('status')
plt.ylabel('time')
plt.show()


# In[72]:


plt.figure(figsize=(8, 5))
sns.countplot(data = df_melanoma, y='ulcer',hue= 'sex', palette='bright')
plt.title('Count plot of sex by ulcer')
plt.xlabel('sex')
plt.ylabel('ulcer')
plt.show()


# In[68]:


#Separate into column subplots based on Sex category
sns.catplot(y = 'status', data = df_melanoma,
             kind = 'count',
              col = 'sex')
plt.show()


# In[75]:


plt.figure(figsize=(8, 5))
sns.boxplot(x='ulcer', y='time', hue='status', data=df_melanoma)
plt.title('Ulceration Impact on Survival')
plt.xlabel('Ulcer')
plt.ylabel('Time')
plt.legend(title='Status', loc='upper left')
plt.show()


# ### Hypothesis T-Test

# Hypothesis Formulation:
#  
#  Null Hypothhesis(H0):There is no significant difference in survival times between patients with ulceration and non-ulceration tumors.
#  Alternative Hypothesis(H1):There is a significant difference in survival times between patients with ulceration and non-ulceration tumors.
#  
#  significance level alpha
#  alpha = 0.05

# In[88]:


# Subset data for patients with ulceration and non-ulceration tumors
ulceration = df_melanoma[df_melanoma['ulcer'] == 'present']['time']
non_ulceration= df_melanoma[df_melanoma['ulcer'] == 'absent']['time']


# In[89]:


ulceration.head()


# In[90]:


non_ulceration.head()


# In[99]:


# Conducting independent samples t-test
stats.ttest_ind(ulceration,non_ulceration,equal_var=False)

# Conduct independent samples p-value
t_statistic,p_value=stats.ttest_ind(ulceration,non_ulceration,equal_var=False)
print('t_statistic:',t_statistic)
print()
print('p_value:',p_value)


# ###### Decision Rule: Since the p-value is less than the alpha we reject the null hypothesis 

# ### Conclusion: There is a significant difference in survival times between patients with ulceration and non-ulceration tumors

# In[ ]:




