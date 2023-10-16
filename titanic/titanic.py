#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 17:23:27 2023

@author: mel
"""

# Data Manipulation Libraries
import numpy as np
import pandas as pd

import seaborn as sns
sns.set()

import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

# Importando os arquivos 
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# Contando nulos
#print(train_data.isnull().sum())
#print(test_data.isnull().sum())

# Para plotar a quantidade de mortos x sobreviventes dado a feature
def barChart(feature):
  surv = train_data[train_data['Survived']==1][feature].value_counts()
  dead = train_data[train_data['Survived']==0][feature].value_counts()
  df = pd.DataFrame([surv, dead])
  df.index = ['Survived', 'Dead']
  df.plot(kind = 'bar', stacked=True, figsize=(8,4))

# barCart('Sex')
# sns.boxplot(x='Age', data = train_data)
# sns.stripplot(x='Age', data = train_data, color = 'black')

# sns.boxplot(x='Pclass', data = train_data)
# sns.stripplot(x='Pclass', data = train_data, color = 'black')

 ######### PRÉ-PROCESSAMENTO #########

# Pegando apenas o título no nome
data = [train_data, test_data]
for dataset in data :
  dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False) # extract only the title of each name record 
  
# print(train_data['Title'].value_counts())
#train_data['Title_encoded'] = encoder.fit_transform(train_data['Title'])
#train_data.drop('Title', axis=1, inplace=True)

# Mapeando os títulos
titleMapping = {"Mr": 0, "Miss": 1,"Mrs": 2, "Master": 3, "Col" : 3, "Rev": 3, "Ms" : 3, "Dr" : 3, "Dona" : 3,
                 "Mlle":3, "Countess" :3, "Capt":3, "Jonkheer":3, "Don":3, "Mme":3, "Lady":3, "Sir":3, "Major" :3}
for dataset in data :
  dataset['Title'] = dataset['Title'].map(titleMapping)
    
# Indexando com ID e dropando a feature Name
train_data.set_index("PassengerId", inplace=True)
train_data.drop('Name', axis =1, inplace= True)
test_data.set_index("PassengerId", inplace=True)
test_data.drop('Name', axis =1, inplace= True)


# Codificando o sexo
train_data['Sex_encoded'] = encoder.fit_transform(train_data['Sex'])
train_data.drop('Sex', axis =1, inplace= True)
test_data['Sex_encoded'] = encoder.fit_transform(test_data['Sex'])
test_data.drop('Sex', axis =1, inplace= True)

  
# Preenchendo os valores vazios em idade com a idade média da classe
train_data["Age"].fillna(train_data.groupby('Pclass')["Age"].transform("median"), inplace = True)
test_data["Age"].fillna(test_data.groupby('Pclass')["Age"].transform("median"), inplace = True)

# Dropando a feature Cabin porque está quase toda nula
train_data.drop('Cabin', axis =1, inplace= True)
test_data.drop('Cabin', axis =1, inplace= True)

# Dropando a feature Ticket porque me parece individual
train_data.drop('Ticket', axis =1, inplace= True)
test_data.drop('Ticket', axis =1, inplace= True)


# Codificando o Embarked
train_data['Embarked_encoded'] = encoder.fit_transform(train_data['Embarked'])
train_data.drop('Embarked', axis=1, inplace=True)
test_data['Embarked_encoded'] = encoder.fit_transform(test_data['Embarked'])
test_data.drop('Embarked', axis=1, inplace=True)

# Removendo linhas que tenham Fare ou Embarked features vazias (neste ponto, não deve ter nenhum elemento nulo na tabela) 
#train_data = train_data.dropna(subset=["Embarked_encoded"])
#test_data = test_data.dropna(subset=["Fare"])
#test_data = test_data.dropna(subset=["Embarked_encoded"])

# Dropando Embarked porque me parece inútil
train_data.drop('Embarked_encoded', axis =1, inplace= True)
test_data.drop('Embarked_encoded', axis =1, inplace= True)

# Combinação de recursos SibSp e Parch, sem nenhuma mudança significativa no resultado
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

# Dropando as colunas SibSp e Parch
train_data.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test_data.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# Função para mapear idades para categorias de faixa etária
def categorize_age(age):
    if age < 18:
        return 0
    elif 18 <= age < 65:
        return 1
    else:
        return 2

    
# Melhorou resultado: ADABOOST, KNN(significativamente), ÁRVORE DE DECISÃO
# Piorou resultado: SVM, GRADIENT BOOSTING, RANDOM FOREST
#train_data['AgeGroup'] = train_data['Age'].apply(categorize_age)
#test_data['AgeGroup'] = test_data['Age'].apply(categorize_age)
#train_data.drop('Age', axis=1, inplace=True)
#test_data.drop('Age', axis=1, inplace=True)


# Preenchendo os valores vazios em fare com a média da classe
train_data["Fare"].fillna(train_data.groupby('Pclass')["Fare"].transform("median"), inplace = True)
test_data["Fare"].fillna(test_data.groupby('Pclass')["Fare"].transform("median"), inplace = True)


 ######### FIM PRÉ-PROCESSAMENTO #########

# Correlação de variáveis
train_corr = train_data.corr()
plt.figure(figsize= (6,6))
sns.heatmap(train_corr,square=True, fmt='.2f', annot=True)

# Separando os dataset de features e classes
X = train_data.drop(columns = ['Survived'])
X.head()
y = train_data['Survived']

# Dividindo o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =.2, random_state = 0)

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
scoring = 'accuracy'

# Árvore de decisão
print("------------ ÁRVORE DE DECISÃO ------------\n")
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

print("Prediction:    ", list(np.round(clf.predict(X_train[:20]))))
print("Actual_result: ", list(y_train[:20]))

score = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print("\nAccuracy rate: ", round(score.mean()*100, 2))

y_pred = clf.predict(X_test)
clf_rmae = mean_absolute_error(y_test, y_pred)
print("Error rate: %.2f\n" %clf_rmae)
print("-------------------------------------------")


# KNN
print("\n\n------------------- KNN -------------------\n")
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)

print("Prediction:    ", list(np.round(knn.predict(X_train[:20]))))
print("Actual_result: ", list(y_train[:20]))

score = cross_val_score(knn, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print("\nAccuracy rate: ", round(score.mean()*100, 2))

y_pred = knn.predict(X_test)
knn_rmae = mean_absolute_error(y_test, y_pred)
print("Error rate: %.2f\n" %knn_rmae)
print("-------------------------------------------")


# Random Forest
print("\n\n------------   RANDOM FOREST   ------------\n")
rf = RandomForestClassifier(n_estimators=5)
rf.fit(X_train, y_train)

print("Prediction:    ", list(np.round(rf.predict(X_train[:20]))))
print("Actual_result: ", list(y_train[:20]))

score = cross_val_score(rf, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print("\nAccuracy rate: ", round(score.mean()*100, 2))

y_pred = rf.predict(X_test)
rf_rmae = mean_absolute_error(y_test, y_pred)
print("Error rate: %.2f\n" %rf_rmae)
print("-------------------------------------------")


# Logistic Regression
print("\n\n------------LOGISTIC REGRESSION------------\n")
lr = LogisticRegression(max_iter=134)
lr.fit(X_train, y_train)

print("Prediction:    ", list(np.round(lr.predict(X_train[:20]))))
print("Actual_result: ", list(y_train[:20]))

score = cross_val_score(lr, X, y, cv=5, scoring=scoring)
print("\nAccuracy rate: ", round(score.mean()*100, 2))

y_pred = lr.predict(X_test)
lr_rmae = mean_absolute_error(y_test, y_pred)
print("Error rate: %.2f\n" %lr_rmae)
print("-------------------------------------------")


# Gradient Boosting
print("\n\n------------ GRADIENT BOOSTING ------------\n")
gb = GradientBoostingClassifier(n_estimators=100, random_state=0)
gb.fit(X_train, y_train)

print("Prediction:    ", list(np.round(gb.predict(X_train[:20]))))
print("Actual_result: ", list(y_train[:20]))

score = cross_val_score(gb, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print("\nAccuracy rate: ", round(score.mean() * 100, 2))

y_pred = gb.predict(X_test)
gb_rmae = mean_absolute_error(y_test, y_pred)
print("Error rate: %.2f\n" % gb_rmae)
print("-------------------------------------------")


# AdaBoost
print("\n\n--------------- ADABOOST ---------------\n")
ab = AdaBoostClassifier(n_estimators=100, random_state=0)
ab.fit(X_train, y_train)

print("Prediction:    ", list(np.round(ab.predict(X_train[:20]))))
print("Actual_result: ", list(y_train[:20]))

score = cross_val_score(ab, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print("\nAccuracy rate: ", round(score.mean() * 100, 2))

y_pred = ab.predict(X_test)
ab_rmae = mean_absolute_error(y_test, y_pred)
print("Error rate: %.2f\n" % ab_rmae)
print("-------------------------------------------")


# Support Vector Machine (SVM)
print("\n\n------------ SUPPORT VECTOR MACHINE ------------\n")
svm = SVC(C=1.0, kernel='rbf', random_state=0)
svm.fit(X_train, y_train)

print("Prediction:    ", list(np.round(svm.predict(X_train[:20]))))
print("Actual_result: ", list(y_train[:20]))

score = cross_val_score(svm, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print("\nAccuracy rate: ", round(score.mean() * 100, 2))

y_pred = svm.predict(X_test)
svm_rmae = mean_absolute_error(y_test, y_pred)
print("Error rate: %.2f\n" % svm_rmae)
print("-------------------------------------------")


prediction = gb.predict(test_data)
submission = pd.DataFrame({'PassengerId': test_data.index, 'Survived': prediction})
submission.to_csv('submission.csv', index=False)
