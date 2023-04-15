import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
df = pd.read_csv('fraud_detection.csv')
df.head()
# Now, let’s have a look at whether this dataset has any null values or not
df.isnull().sum()
df.describe()
df.info()
df.shape
# Exploring transaction type
df.type.value_counts()
sns.barplot(x = df.type.value_counts().index, y = df.type.value_counts().values)
plt.show()
labels = ['CASH_OUT','PAYMENT','CASH_IN','TRANSFER', 'DEBIT']
values = df["type"].value_counts()
plt.title('Distribution of Transaction Type')
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.legend(loc='upper left', fontsize="small")
df.nlargest(10, 'amount')
df.isFraud.value_counts()
sns.barplot(x = df.isFraud.value_counts().index, y = df.isFraud.value_counts().values)
plt.show()
labels = ['isFraud','notFraud']
values = df["isFraud"].value_counts()
plt.title('Fraud Data')
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.legend(loc='upper left', fontsize="small")
#correlation between the features of the data with the isFraud column
# Checking correlation
correlation = df.corr()
print(correlation["isFraud"].sort_values(ascending=False))
sns.heatmap(df.corr(), annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(20,20)
plt.show()
