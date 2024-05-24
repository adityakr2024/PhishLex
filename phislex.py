#!/usr/bin/env python
# coding: utf-8

# Load libraries

# In[45]:


import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn import metrics
import re
import warnings
warnings.filterwarnings('ignore')


# Load URLs data

# In[2]:


df = pd.read_csv('urldataset.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# Extract Features

# In[5]:


# Length of URL
df['url_length'] = df['url'].apply(lambda x: len(x))

# URL has http
df['url_has_http//'] = df['url'].apply(lambda x: 1 if 'http://' in x else 0)

# URL has https
df['url_has_https//'] = df['url'].apply(lambda x: 1 if 'https://' in x else 0)

# URL has num
df['url_contains_num'] = df['url'].apply(lambda url: 1 if re.search(r'\d', url) else 0)

# Number of dots
df['url_num_dots'] = df['url'].apply(lambda x: x.count('.'))

# Number of slashes
df['url_num_slashes'] = df['url'].apply(lambda x: x.count('/'))

# Number of Dash
df['number_of_dash'] = df['url'].apply(lambda x: x.count('-'))

# @ Symbol
df['has_at_symbol'] = df['url'].apply(lambda x: 1 if '@' in x else 0)

# Tilde Symbol
df['has_tilde_symbol'] = df['url'].apply(lambda x: 1 if '~' in x else 0)

# Number of Underscore
df['number_of_underscore'] = df['url'].apply(lambda x: x.count('_'))

# Number of NumPercent
df['number_of_numpercent'] = df['url'].apply(lambda x: x.count('%'))

# Number of Ampersand
df['number_of_ampersand'] = df['url'].apply(lambda x: x.count('&'))

# Number of Hash
df['number_of_hash'] = df['url'].apply(lambda x: x.count('#'))

# URL contains IP address
df['url_contains_ip'] = df['url'].apply(lambda url: 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url) else 0)


# In[6]:


df.head()


# In[7]:


df.columns


# In[8]:


df1 = df.drop(['url'], axis = 1)


# In[9]:


df1.head()


# In[10]:


df1.shape


# In[11]:


df1.info()


# In[12]:


df1.columns


# In[13]:


df1['result'].value_counts().plot(kind='pie',autopct='%1.2f%%')
plt.title("Phishing Count")
plt.show()


# In[14]:


#Independent Variables
X = df[['url_length', 'url_has_http//', 'url_has_https//', 'url_contains_num', 'url_num_dots', 'url_num_slashes', 'number_of_dash', 'has_at_symbol', 'has_tilde_symbol', 'number_of_underscore', 'number_of_numpercent', 'number_of_ampersand', 'number_of_hash', 'url_contains_ip']]

#Dependent Variable
y = df['result']


# Train test split

# In[15]:


from imblearn.over_sampling import SMOTE

X_sample, y_sample = SMOTE().fit_resample(X, y.values.ravel())

X_sample = pd.DataFrame(X_sample)
y_sample = pd.DataFrame(y_sample)

print("Size of X-sample :", X_sample.shape)
print("Size of y-sample :", y_sample.shape)


# In[16]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size = 0.2, random_state = 2529)
print("Shape of X_train: ", X_train.shape)
print("Shape of X_valid: ", X_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_valid: ", y_test.shape)


#  XGBoost Classifier

# In[17]:


#XGBoost Classifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train,y_train)

xg_predictions = xgb_model.predict(X_test)
accuracy_score(y_test,xg_predictions)


# In[18]:


cm = pd.DataFrame(confusion_matrix(y_test,xg_predictions))
cm.columns = ['Predicted 0', 'Predicted 1']
cm = cm.rename(index = {0:'Actual 0',1:'Actual 1'})
cm


# Random Forest

# In[19]:


#Random Forest
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

rfc_predictions = rfc.predict(X_test)
accuracy_score(y_test, rfc_predictions)


# In[20]:


cm = pd.DataFrame(confusion_matrix(y_test,rfc_predictions))
cm.columns = ['Predicted 0', 'Predicted 1']
cm = cm.rename(index = {0:'Actual 0',1:'Actual 1'})
cm


# Logistic Regression

# In[21]:


#Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train,y_train)

log_predictions = log_model.predict(X_test)
accuracy_score(y_test,log_predictions)


# In[22]:


cm_df = pd.DataFrame(confusion_matrix(y_test,log_predictions))
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index = {0:'Actual 0',1:'Actual 1'})
cm_df


# K-Nearest Neighbors Classifier

# In[23]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
Knn=knn.fit(X_train,y_train)
k_predict=Knn.predict(X_test)
accuracy_score(y_test,k_predict)


# In[24]:


confusion_matrix(y_test,k_predict)


# Naive Bayes Classifier

# In[25]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
Nb=nb.fit(X_train,y_train)
n_predict=nb.predict(X_test)
accuracy_score(y_test,n_predict)


# In[26]:


confusion_matrix(y_test,n_predict)


# Decision Trees Classifier

# In[30]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=30)
Dt=dt.fit(X_train, y_train)
d_predict=Dt.predict(X_test)
accuracy_score(y_test,d_predict)


# In[31]:


confusion_matrix(y_test,d_predict)


# Gradient Boosting Classifier

# In[32]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)
Gbc=gbc.fit(X_train,y_train)
g_predict=Gbc.predict(X_test)
accuracy_score(y_test,g_predict)


# In[33]:


confusion_matrix(y_test,g_predict)


# In[40]:


#Overall Accuracy table
import numpy as np
model = np.array(['XGBClassifier', 'Random Forest', 'Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes', 'Decision Trees', 'Gradient Boosting'])
scr = np.array([accuracy_score(y_test,xg_predictions)*100, accuracy_score(y_test, rfc_predictions)*100, accuracy_score(y_test,log_predictions)*100, accuracy_score(y_test,k_predict)*100, accuracy_score(y_test,n_predict)*100, accuracy_score(y_test,d_predict)*100, accuracy_score(y_test,g_predict)*100])
tbl = pd.DataFrame({"Model": model,"Accuracy Score": scr})
tbl


# In[ ]:


df1.shape


# In[50]:


def predict_url(url):
  url_length = len(url)
  url_has_http = 1 if 'http://' in url else 0
  url_has_https = 1 if 'https://' in url else 0
  url_contains_num = 1 if re.search(r'\d', url) else 0
  url_num_dots = url.count('.')
  url_num_slashes = url.count('/')
  number_of_dash = url.count('_')
  has_at_symbol = 1 if '@' in url else 0
  has_tilde_symbol = 1 if '~' in url else 0
  number_of_underscore = url.count('_')
  number_of_numpercent = url.count('%')
  number_of_ampersand = url.count('&')
  number_of_hash = url.count('#')
  url_contains_ip = 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url) else 0

  # Create a feature vector
  features = [url_length, url_has_http, url_has_https, url_contains_num, url_num_dots,
              url_num_slashes, number_of_dash, has_at_symbol, has_tilde_symbol,
              number_of_underscore, number_of_numpercent, number_of_ampersand,
              number_of_hash, url_contains_ip]

  # Convert the feature vector to a NumPy array
  features = np.array(features).reshape(1, -1)

  # Predict the URL
  # prediction = xgb_model.predict(features)[0]
  prediction = rfc.predict(features)[0]
  # prediction = log_model.predict(features)[0]
  # prediction = Knn.predict(features)[0]
  # prediction = nb.predict(features)[0]
  # prediction = Dt.predict(features)[0]
  # prediction = Gbc.predict(features)[0]



  return prediction

# Get the user input URL
print("Input Format: www.example.com")
url = input("Enter a URL: ")

# Predict the URL
prediction = predict_url(url)

# Print the prediction
if prediction == 1:
  print("The URL is malicious.")
else:
  print("The URL is legitimate.")


# In[ ]:




