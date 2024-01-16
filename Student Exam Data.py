#!/usr/bin/env python
# coding: utf-8

# ## Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ## Data Loading and Preprocessing

# In[2]:


pd.set_option('display.max_columns',90)
data = pd.read_csv('student_exam_data.csv')
data


# In[3]:


data.describe()


# In[4]:


data.isnull().sum()


# In[5]:


data.dtypes


# In[6]:


data.corr()['Pass/Fail']


# In[7]:


data.columns


# ## Checking Multicollinearity between independent variables with VIF

# In[8]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data[['Study Hours', 'Previous Exam Score']]

vif=pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
vif


# ## Outliear Checking

# In[9]:


for i in data[['Study Hours', 'Previous Exam Score']]:
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[10]:


data = data.reset_index()


# In[11]:


data.head()


# In[12]:


data.drop('index', axis=1, inplace=True)


# In[13]:


data


# ## WOE Transformation for Logistic Regression

# In[14]:


ranges=[-np.inf, data['Study Hours'].quantile(0.25), data['Study Hours'].quantile(0.5), data['Study Hours'].quantile(0.75), np.inf]
data['Study_Hours_category'] = pd.cut(data['Study Hours'], bins=ranges)
data


# In[15]:


grouped = data.groupby(['Study_Hours_category', 'Pass/Fail'])['Pass/Fail'].count().unstack().reset_index()
grouped


# In[16]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['Study_Hours_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[17]:


data = data.merge(grouped[['Study_Hours_category', 'Study_Hours_woe']], how = 'left', on = 'Study_Hours_category')
data


# In[18]:


ranges=[-np.inf, data['Previous Exam Score'].quantile(0.25), data['Previous Exam Score'].quantile(0.5), data['Previous Exam Score'].quantile(0.75), np.inf]
data['Previous_Exam_Score_category'] = pd.cut(data['Previous Exam Score'], bins=ranges)
data


# In[19]:


grouped = data.groupby(['Previous_Exam_Score_category', 'Pass/Fail'])['Pass/Fail'].count().unstack().reset_index()
grouped


# In[20]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['Previous_Exam_Score_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[21]:


data = data.merge(grouped[['Previous_Exam_Score_category', 'Previous_Exam_Score_woe']], how = 'left', on = 'Previous_Exam_Score_category')
data


# In[22]:


data_woe = data[['Study_Hours_woe', 'Previous_Exam_Score_woe', 'Pass/Fail']]
data_woe


# In[23]:


data_woe[np.isinf(data_woe)] = 1.5 # Here I set every inf values to 1.5, since aftee woe transformation I got inf values in my dataset. 


# In[24]:


data_woe


# ## Data splitting and Modelling

# In[25]:


X = data_woe[['Study_Hours_woe', 'Previous_Exam_Score_woe']]
y = data_woe['Pass/Fail']


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# ## Evaluate Function to see Accuracy Results after ML applying
# 

# In[29]:


def evaluate(model, X_test, y_test):
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:,1]
    
    roc_score_test = roc_auc_score(y_test, y_prob_test)
    gini_score_test = roc_score_test*2-1
    
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:,1]
    
    roc_score_train = roc_auc_score(y_train, y_prob_train)
    gini_score_train = roc_score_train*2-1
    
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred_test)
    
    accuracy_score_test = metrics.accuracy_score(y_test, y_pred_test)
    accuracy_score_train = metrics.accuracy_score(y_train, y_pred_train)
    
    print('Model Performance:')

    print('Gini Score for Test:', gini_score_test*100)
    
    print('Gini Score for Train:', gini_score_train*100)
    
    print('Accuracy Score for Test:', accuracy_score_test*100)
    
    print('Accuracy Score for Train:', accuracy_score_train*100)
    
    print('Confusion Matrix:', confusion_matrix)


# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


lr = LogisticRegression()


# In[32]:


lr.fit(X_train, y_train)


# In[33]:


result = evaluate(lr, X_test, y_test)


# In[34]:


y_prob = lr.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[35]:


from sklearn.tree import DecisionTreeClassifier


# In[36]:


dtc = DecisionTreeClassifier()


# In[37]:


dtc.fit(X_train, y_train)


# In[38]:


result = evaluate(dtc, X_test, y_test)


# In[39]:


y_prob = dtc.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[40]:


from sklearn.ensemble import RandomForestClassifier


# In[41]:


rfc = RandomForestClassifier()


# In[42]:


rfc.fit(X_train, y_train)


# In[43]:


result = evaluate(rfc, X_test, y_test)


# In[44]:


y_prob = rfc.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[45]:


from lightgbm import LGBMClassifier


# In[46]:


lgbm = LGBMClassifier()


# In[47]:


lgbm.fit(X_train, y_train)


# In[48]:


result = evaluate(lgbm, X_test, y_test)


# In[49]:


y_prob = lgbm.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[50]:


from xgboost import XGBClassifier


# In[51]:


xgboost = XGBClassifier()


# In[52]:


xgboost.fit(X_train, y_train)


# In[53]:


result = evaluate(xgboost, X_test, y_test)


# In[54]:


y_prob = xgboost.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[55]:


from catboost import CatBoostClassifier


# In[56]:


catboost = CatBoostClassifier()


# In[57]:


catboost.fit(X_train, y_train)


# In[58]:


result = evaluate(catboost, X_test, y_test)


# In[59]:


y_prob = catboost.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[61]:


variables= []
train_Gini=[]
test_Gini=[]

for i in X_train.columns:
    X_train_single=X_train[[i]]
    X_test_single=X_test[[i]]
    
    catboost.fit(X_train_single, y_train)
    y_prob_train_single=catboost.predict_proba(X_train_single)[:, 1]
    
    
    roc_prob_train=roc_auc_score(y_train, y_prob_train_single)
    gini_prob_train=2*roc_prob_train-1
    
    
    catboost.fit(X_test_single, y_test)
    y_prob_test_single=catboost.predict_proba(X_test_single)[:, 1]
    
    
    roc_prob_test=roc_auc_score(y_test, y_prob_test_single)
    gini_prob_test=2*roc_prob_test-1
    
    
    variables.append(i)
    train_Gini.append(gini_prob_train)
    test_Gini.append(gini_prob_test)
    

df = pd.DataFrame({'Variable': variables, 'Train Gini': train_Gini, 'Test Gini': test_Gini})

df= df.sort_values(by='Test Gini', ascending=False)

df   


# In[ ]:




