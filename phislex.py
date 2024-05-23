import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn import metrics
import re
import warnings
warnings.filterwarnings('ignore')

"""Loading the data"""

df = pd.read_csv('urldataset.csv')

df.head()

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

df.head()

df = df.drop(['url'], axis = 1)

df.head()

df['result'].value_counts()

df.shape

df.columns

df.info()

df.describe()

df.nunique()

df.isnull().sum()

df.describe().T

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.duplicated().sum()

df.shape

clmns=['result','url_has_http//','url_has_https//']
for i in clmns:
    print(f"The Value counts of {i} :")
    print(df[i].value_counts().to_string(),'\n')

df['result'].value_counts().plot(kind='pie',autopct='%1.2f%%')
plt.title("Phishing Count")
plt.show()

"""0 = Not phis(legitmate)
1 = phising
"""

sns.distplot( a=df["result"], hist=True, kde=False, rug=False )

sns.kdeplot(df['result'])

plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), annot=True)
plt.show()

df.corr()

for i in df.columns:
    if type(df[i][0])!=str:
        sns.boxplot(df[i])
        plt.title(i)
        plt.show()

# Z-score method
z_scores = np.abs((df - df.mean()) / df.std())
outliers = df[z_scores > 3]

# IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))]

#Handle outliers
# Remove outliers
df_cleaned = df.drop(outliers.index)

# Replace outliers with median
df_cleaned = df.copy()
df_cleaned=np.where(z_scores>3,df.median(),df)

# Apply log transformation
df_cleaned = np.log1p(df+0.0001)

# Validate the results
print(df_cleaned.describe())

df_cleaned.hist()
plt.show()

# Verify the results using box plots

plt.figure(figsize=(10, 6))


# Original data

plt.subplot(1, 2, 1)

plt.boxplot(df.values)

plt.title('Original Data')


# Cleaned data

plt.subplot(1, 2, 2)

plt.boxplot(df_cleaned.values)

plt.title('Cleaned Data')

plt.show()

X = df.drop(["result"],axis =1)
y = df["result"]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
clmn_names=X.columns
scaled=scaler.fit_transform(X)
features=pd.DataFrame(scaled,columns=clmn_names)
features

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

X_train.shape

y_train.shape

X_test.shape

y_test.shape

from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy=1)
X_train,y_train=smote.fit_resample(X_train,y_train)
pd.DataFrame(y_train).value_counts()

"""Logistic Regression"""

from sklearn.linear_model import LogisticRegression
# instantiate the model
log = LogisticRegression()
# fit the model
Log=log.fit(X_train,y_train)
predict=Log.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

accuracy_score(y_test,predict)

confusion_matrix(y_test,predict)

print(classification_report(y_test,predict))

"""K-Nearest Neighbors : Classifier"""

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
Knn=knn.fit(X_train,y_train)
k_predict=Knn.predict(X_test)

accuracy_score(y_test,k_predict)

confusion_matrix(y_test,k_predict)

print(classification_report(y_test,k_predict))

"""Naive Bayes : Classifier"""

# Naive Bayes Classifier Model
from sklearn.naive_bayes import GaussianNB

# instantiate the model
nb = GaussianNB()

# fit the model
Nb=nb.fit(X_train,y_train)
n_predict=nb.predict(X_test)

accuracy_score(y_test,n_predict)

confusion_matrix(y_test,n_predict)

print(classification_report(y_test,n_predict))

"""Decision Trees : Classifier"""

# Decision Tree Classifier model
from sklearn.tree import DecisionTreeClassifier

# instantiate the model
dt = DecisionTreeClassifier(max_depth=30)

# fit the model
Dt=dt.fit(X_train, y_train)
d_predict=Dt.predict(X_test)

accuracy_score(y_test,d_predict)

confusion_matrix(y_test,d_predict)

print(classification_report(y_test,d_predict))

"""Random Forest : Classifier"""

# Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier

# instantiate the model
forest = RandomForestClassifier(n_estimators=10)

# fit the model
Forest=forest.fit(X_train,y_train)
f_predict=Forest.predict(X_test)

accuracy_score(y_test,f_predict)

confusion_matrix(y_test,f_predict)

print(classification_report(y_test,f_predict))

from sklearn.model_selection import cross_val_score
cv=cross_val_score(Forest,X_train,y_train,cv=11)
np.mean(cv)

"""Gradient Boosting Classifier

"""

# Gradient Boosting Classifier Model
from sklearn.ensemble import GradientBoostingClassifier

# instantiate the model
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

# fit the model
Gbc=gbc.fit(X_train,y_train)
g_predict=Gbc.predict(X_test)

accuracy_score(y_test,g_predict)

confusion_matrix(y_test,g_predict)

print(classification_report(y_test, g_predict))

"""Multi-layer Perceptron classifier"""

# Multi-layer Perceptron Classifier Model
from sklearn.neural_network import MLPClassifier

# instantiate the model
mlp = MLPClassifier()
#mlp = GridSearchCV(mlpc, parameter_space)

# fit the model
Mlp=mlp.fit(X_train,y_train)
m_predict=Mlp.predict(X_test)

accuracy_score(y_test,m_predict)

confusion_matrix(y_test,m_predict)

print(classification_report(y_test,m_predict))

"""Support Vector Machine : Classifier"""

# Support Vector Classifier model
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'gamma': [0.1],'kernel': ['rbf','linear']}

svc = GridSearchCV(SVC(), param_grid)

# fitting the model for grid search
Svc=svc.fit(X_train, y_train)
s_predict=Svc.predict(X_test)

accuracy_score(y_test,s_predict)

confusion_matrix(y_test,s_predict)

print(classification_report(y_test, s_predict))

"""Comparision of Models"""

# Instantiate the models
logistic_regression = LogisticRegression()
knn = KNeighborsClassifier()
naive_bayes = GaussianNB()
decision_tree = DecisionTreeClassifier(max_depth=30)
random_forest = RandomForestClassifier(n_estimators=10)
gradient_boosting = GradientBoostingClassifier(max_depth=4, learning_rate=0.7)
mlp = MLPClassifier()
svc = SVC()

# Train and predict for each model
models = [
    ("Logistic Regression", logistic_regression),
    ("K-Nearest Neighbors", knn),
    ("Naive Bayes", naive_bayes),
    ("Decision Tree", decision_tree),
    ("Random Forest", random_forest),
    ("Gradient Boosting", gradient_boosting),
    ("Multi-Layer Perceptron", mlp),
    ("Support Vector", svc)
]

for model_name, model in models:
    print(f"Model: {model_name}")

    # Fit the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"Classification Report:\n{classification}")
    print("-" * 50)

"""Testing the Model"""

X_train.head(20)

X_train.iloc[480]

Gbc.predict([[36,	0,	1,	0,	2,	5,	0,	0,	0,	0,	0,	0,	0,	0]])

"""Storing Best Model"""

import pickle

# dump information to that file
pickle.dump(Gbc, open('Phishing_model.pkl', 'wb'))

# prompt: write a code to predict user input URL according to our model

def predict_url(url):
  # Extract features from the URL
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
  prediction = Gbc.predict.predict(features)[0]

  # Return the prediction
  return prediction

# Get the user input URL
url = input("Enter a URL: ")

# Predict the URL
prediction = predict_url(url)

# Print the prediction
if prediction == 1:
  print("The URL is malicious.")
else:
  print("The URL is legitimate.")

