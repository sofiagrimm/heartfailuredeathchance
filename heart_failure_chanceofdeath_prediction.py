#import libraries
import pandas as pd
import seaborn as sns
import sklearn

import os
os.environ['KAGGLE_USERNAME'] = 'sofialol'
os.environ['KAGGLE_KEY'] = '7d3db3deabbe2a8aeb0480aa3e1e8e48'

#download dataset
! kaggle datasets download -d andrewmvd/heart-failure-clinical-data

#unzip file
! unzip /content/heart-failure-clinical-data.zip

#load data on dataframe
df = pd.read_csv('/content/heart_failure_clinical_records_dataset.csv')

#display dataframe
df.head()

df.shape

df.isna().sum()

df.dropna(axis=1, inplace=True)

df.shape

df['DEATH_EVENT'].value_counts()

#independent & dependent split
X=df.iloc[1:,:12].values
Y=df.iloc[1:,12].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression as LR
classifier = LR()
classifier.fit(X_train, Y_train)

#input instead of x-test & put into array to see how likely it is for someone to die (function)

predictions = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(Y_test, predictions)
print(cm)
sns.heatmap(cm,annot=True)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))

print(Y_test)

print(predictions)
