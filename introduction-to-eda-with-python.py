#!/usr/bin/env python
# coding: utf-8

# # Introduction to Exploratory Data Analysis with Python
# 
# ![Fight Club](https://images.gr-assets.com/hostedimages/1445796906ra/16703819.gif)
# 
# The first rule of Data science club is: you don't call it *Exploratory Data Analysis*, instead - it is **EDA**
# 
# Before we get into EDA, Let's begin with **Data Science**
# 
# 
# 

# ## What is Data Science?
# 
# 
# "What is Data Science?  There are now like, you know, a billion venn diagrams showing you what data science is.  But to me I think the definition is pretty simple.  Whenever you're struggling with data, trying to understand what's going on with data, whenever you're trying to turn that **raw data into insight and understanding and discoveries**.  I think that's **Data Science**." - **[Hadley Wickham](https://en.wikipedia.org/wiki/Hadley_Wickham)**
# 
# 
# ## Data Analysis/Science Framework - Kind of!
# 
# ![DS Framework](https://preview.ibb.co/nBBUK9/datascience_framework_hadley.png)
# 

# 
# ## EDA
# 
# According to [Wikipedia](https://en.wikipedia.org/wiki/Exploratory_data_analysis), EDA is an approach to **analyzing data sets** to **summarize their main characteristics**, often with **visual methods**.

# ## The Journey begins
# 
# ### with importing necessary Libraries

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#import os
#print(os.listdir("../input"))


# ### and reading Input Data

# In[3]:


survey = pd.read_csv("survey_results_public.csv")


# Similar to Real Estate where usually the very first question is **How many square feet is the house/property?**  - we will start asking **What is the dimension (Number of Rows and Columns) of the input data?**

# In[4]:


survey.shape


# ### Can we get a top view? 

# In[5]:


survey.head()


# In[6]:


[column for column in survey.columns]


# Once we get an understanding of what's in the Data, It's always a good practice to formulate some initial set of questions you'd like to answer from Data and then work towards them. The answers of those questions are inturn called **Insights/Findings** which could further help in business decisions or similar outcomes

# # Questions:
# 
# 1. This being Survey data, How many respondents are there?
# 2. How many male and female respondents are there?
# 3. What can we understand about the Age of the respondents?

# In[7]:


len(survey.index)


# Similar to MS Excel **pivot** (or Similar to GROUP BY IN SQL), Python Pandas has `groupby()`
# 

# In[8]:


survey.groupby("Gender").size()


# Wait, This is not possible right? And, this is the part where a Data Scientist realizes the importance of **Data Cleaning**.
# 
# ![Data Cleaning](https://image.slidesharecdn.com/trifactakueckerlogisticswebinarv1-171212235026/95/how-kuecker-logistics-onboards-customers-5-times-faster-14-638.jpg?cb=1513122713)

# 
# 

# Let's take only those rows where `Gender` value is either `Male` or `Female` and then repeat what we did above.

# In[9]:


survey[survey["Gender"].isin(["Male","Female"])].groupby("Gender").size()


# That was okay, but too much of numbers to undertstand how the data split is. That's when Percentage can help us.
# 
# Below is how you can write step by step (when you start with Pandas)

# In[10]:


(survey[survey["Gender"].isin(["Male","Female"])].groupby("Gender").size()/len(survey[survey["Gender"].isin(["Male","Female"])].index))*100


# Below is how you can write the same - short and sweet (when you know more Pandas)

# In[11]:


survey[survey["Gender"].isin(["Male","Female"])].groupby("Gender").size().transform(lambda x: (x/sum(x))*100)


# Well, That kind of helps. 

# In[12]:


survey["Age"][1:10]


# That seems to have got something weird `NaN` - which is what we call **Missing Values**.  
# 
# Let us see how many such missing values are there in `Age`

# In[13]:


print("Total number of rows is ", len(survey["Age"]), " and \nNumber of missing values is ", np.count_nonzero(survey["Age"].isnull()))


# Anyways, `groupby()` by default ignores them!

# In[14]:


survey.groupby('Age').size()


# Well, Now that's a lot of numbers and Didn't we see at the first that EDA is also about communicating insights visually? 

# In[15]:


survey.groupby('Age').size()


# In[16]:


sns.countplot(x="Age", data=survey)


# Doesn't it look ugly? Kind of! 
# 
# Let's add some flavor to the above plot!

# That's a theme!

# In[17]:


plt.style.use('ggplot')


# In[18]:


sns.countplot(x="Age", data=survey)


# Improved, Still cluttered! Let's see if we can rotate them!

# In[19]:


sns.countplot(y="Age", data=survey)


# Now, that's a lot better! But still our plot doesn't have any title. (It's not naked completely, but half-naked)

# In[20]:


sns.countplot(y="Age", data=survey)
plt.title("Age Count of StackOverflow Survey Data")


# In[21]:


sns.countplot(y="Age", data=survey).set_title("Age Count of StackOverflow Survey Data")


# In[22]:


sns.set(rc={'figure.figsize':(12,8)})


# In[23]:


sns.countplot(y="Age", data=survey).set_title("Age Count of StackOverflow Survey Data")


# What if we want to see `Age` along with another variable? Let's say `JobSatisfaction`?

# In[24]:


sns.countplot(x="Age", hue = "JobSatisfaction", data=survey).set_title("Age Count of StackOverflow Survey Data")


# Can we add one more dimension to our plot?

# In[25]:


filtered = survey[survey["Gender"].isin(["Male","Female"])].dropna(subset = ['Age'])
sns.set(rc={'figure.figsize':(12,8)})
sns.catplot(y="Age", hue = "JobSatisfaction", col = "Gender", data= filtered, kind="count", height=10, aspect = 0.9)


# ### That's pretty much it!

# ![](https://i.ytimg.com/vi/fylsLOe8ClY/maxresdefault.jpg)

# It's a shame, If had introduced EDA without showing the beautiful `violinplots`. So here it is!

# In[26]:


sns.set(style="whitegrid")
sns.violinplot(x="Gender", y="ConvertedSalary", data=filtered);


# # Now, you might question my definition of "Beautiful". Isn't it? 
# 
# Even though this is aesthetically beautiful (sorry, it is actually), it's also insightfully beautiful because this piece of plot just showed how oiur data is distributed with **Outliers**.
# 
# After all, this is Salary distribution and a few people make a lot of money than a lot of people ;)

# In[27]:


plt.style.use("fivethirtyeight")
sns.violinplot(y="JobSatisfaction", x="ConvertedSalary", data=filtered).set_title("Slightly More Beautiful")


# # Thank you!
