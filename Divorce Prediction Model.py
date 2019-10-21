#!/usr/bin/env python
# coding: utf-8

# Importing Essential Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df = pd.read_excel('divorce.xlsx')
df.head()


# EDA

df.info()


df.describe().T


df['Class'].value_counts()


z = df.drop('Class', 1).values
y = df['Class'].values


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pc = pca.fit_transform(z)
pcdf = pd.DataFrame(pc, columns=['pc 1', 'pc 2'])
finaldf = pd.concat([pcdf, df['Class']], axis=1)
finaldf


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Pricipal Component 1', fontsize = 13)
ax.set_ylabel('Pricipal Component 2', fontsize = 13)
ax.set_title('PCA Visualization', fontsize = 20)
targets = [0, 1]
colors = ['g', 'r']
for target, color in zip(targets, colors):
    indeces = finaldf['Class'] == target
    ax.scatter(finaldf.loc[indeces, 'pc 1'], finaldf.loc[indeces, 'pc 2'], c = color, s= 50 )
ax.legend(['Not-Divorced', 'Divorced'], loc = (1.05, 0.85), fontsize = 18)
ax.grid()


# Finding Correlation

corr = df.corr()
cor = abs(corr['Class'])
imp = cor[cor > 0.5]
imp


plt.figure(figsize=(30,30))
sns.heatmap(df.corr(), annot=  True, cmap = 'RdYlGn')
plt.show()


# Choosing some good models

X = df.drop('Class', 1)
y = df['Class']


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)


# Checking Performance of Each Model

scaler = StandardScaler()
X = scaler.fit_transform(X)
models = [RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
         ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced'), 
         DecisionTreeClassifier(random_state=42, class_weight='balanced'), 
         LogisticRegression(solver='lbfgs', random_state=42, class_weight='balanced')]

classifiers = ['Ramdom Forest', 'Extra Tree', 'Decision Tree', 'Logistic Regression']

for (model, name) in zip(models, classifiers):
    model = model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    errors = (y_pred != y_test).sum()
    print('{} Errors: {} \n'.format(name, errors))
    cv = cross_val_score(model, X, y, cv=5)
    print('CV Mean: {} '.format(np.mean(cv)))
    print('Accuracy Score: {} '.format(accuracy_score(y_test, y_pred)))
    print('ROC_AUC Score: {}  '.format(roc_auc_score(y_test, y_pred)))
    print('Confusion Matrix: \n {} \n'.format(confusion_matrix(y_test, y_pred)))
    print('Classification Report: \n {} \n'.format(classification_report(y_test, y_pred)))


# Making a PipeLine

# Choosing DecisionTree Classifier

model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
scaler = StandardScaler()
pca = PCA(.95)
pipeline = make_pipeline(scaler, pca, model)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
errors = (y_pred != y_test).sum()
print('Errors of "Decision Tree Classifier": {} \n'.format( errors))
cv = cross_val_score(model, X, y, cv=5)
print('CV Mean: {} '.format(np.mean(cv)))
print('Accuracy Score: {} '.format(accuracy_score(y_test, y_pred)))
print('ROC_AUC Score: {}  '.format(roc_auc_score(y_test, y_pred)))
print('Confusion Matrix: \n {} \n'.format(confusion_matrix(y_test, y_pred)))
print('Classification Report: \n {} \n'.format(classification_report(y_test, y_pred)))


# Saving the Model

from joblib import dump
dump(pipeline, 'Divorce Predictor.pkl')
