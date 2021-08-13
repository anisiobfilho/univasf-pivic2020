# PIVIC - "Um modelo computacional para identificação de notícias falsas sobre a Covid-19 no Brasil"
# Code: Machine Learning - Supervised Learning
# Author: Anísio Pereira Batista Filho

##Essentials
import os
import csv
import numpy as np ##Numpy
import pandas as pd ##Pandas
##Sci-kit Learn
###Machine learning algorithms
from xgboost import XGBClassifier, XGBRegressor
from imblearn.pipeline import Pipeline as imblearnPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
##Model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
###Pipeline, vectorizers and preprocessing
from sklearn.pipeline import Pipeline as sklearnPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
###Metrics
from sklearn.metrics import classification_report, accuracy_score
##Utils
import re
import unicodedata
from tqdm.auto import tqdm
import time
import timeit

start = timeit.default_timer()

#Colunas: tweet_text;tweet_text_lower;tweet_text_stemmed;tweet_text_lemmatized;tweet_text_spellchecked;tweet_text_spellchecked_lower;tweet_text_spellchecked_stemmed;tweet_text_spellchecked_lemmatized
##lendo o dataset
df = pd.read_csv("data/data-twitter/training/rotulaçao[iguais]_complete.csv", sep=";")
#X=df[['time_shift', 'region_location']] #'state_location',
X = pd.DataFrame()
#X['tweet_text_stemmed']=df['tweet_text_stemmed'].apply(lambda x: np.str_(x))
X['tweet_text_stemmed']=df.loc[:,'tweet_text_stemmed'].apply(lambda x: np.str_(x))
#X = pd.DataFrame()
#X=df[['tweet_text_stemmed']].astype(str)
#X=df[['time_shift', 'region_location']] #'state_location',
#text_list = df.loc[:,'tweet_text_stemmed'].apply(lambda x: np.str_(x))
#X['tweet_text_stemmed'] = text_list

#X['created_at'] = pd.to_numeric(df.created_at.str.replace('-','').replace(':','').replace('+','').replace(' ',''))
y = pd.DataFrame()
y['label'] = df.loc[:,'label_A']
y.label += 1

##Separando dados de treinamento e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y.label, test_size = 0.25, random_state = 10)

##Pipeline para o MinMaxScaler()
minmax_transformer = sklearnPipeline(steps=[
    ('imputer', MinMaxScaler(feature_range=(0, 1)))
])

##Pipeline para simpleimputer()
num_transformer = sklearnPipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

##Pipeline para OneHotEncoder()
cat_transformer = sklearnPipeline(steps=[
    ('one-hot encoder', OneHotEncoder())
])

##Pipeline para TfidfVectorizer()
tfidf_transfomer = sklearnPipeline(steps=[
    ('tf-idf', TfidfVectorizer())
])

##Compondo os pré-processadores
preprocessor = ColumnTransformer(transformers=[
    #('minmax', minmax_transformer, ['created_at']),
    #('num', num_transformer, ['amount']),
    ('tf-idf', tfidf_transfomer, 'tweet_text_stemmed'),
    #('cat', cat_transformer, ['time_shift', 'region_location'])
    ],
    #remainder='passthrough'
    )

##Criando o modelo usando pipeline
model = imblearnPipeline(steps=[
    ('preprocessor', preprocessor),

    #('oversampler', SMOTE()),
    #('undersampler', RandomUnderSampler()),

    #('decisiontree', DecisionTreeClassifier()),
    #('multinomial-naive-bayes', MultinomialNB()),
    #('svc', SVC())
    #('randomforest', RandomForestClassifier()),

    #('adaboost', AdaBoostClassifier(base_estimator=DecisionTreeClassifier())),
    #('adaboost', AdaBoostClassifier(base_estimator=MultinomialNB())),
    #('adaboost', AdaBoostClassifier(base_estimator=SVC())),
    #('adaboost', AdaBoostClassifier(base_estimator=RandomForestClassifier())),

    #('xgboost', XGBClassifier())
])

##Tunando hiperparâmetros com 10-fold cross-validation e pipelines
#DECISION TREE
#parameters = {  'decisiontree__criterion': ['gini', 'entropy'],
#                'decisiontree__splitter': ['best', 'random'],
#                'decisiontree__max_depth': [9, 10, None],
#                'decisiontree__min_samples_split': [2, 3, 4, 5],
#                'decisiontree__min_samples_leaf': [1, 2, 3],
#                #'decisiontree__min_weight_fraction_leaf': [0.0],
#                #'decisiontree__max_features': [None],
#                'decisiontree__random_state': [1, None],
#                #'decisiontree__max_leaf_nodes': [None],
#                #'decisiontree__min_impurity_decrease': [0.0],
#                #'decisiontree__class_weight': [None],
#                #'decisiontree__ccp_alpha': [0.0]
#            }

#MULTINOMIALNB
#parameters = {  'multinomial-naive-bayes__alpha': [0.01, 0.1, 0.5, 1.0, 10.0],
#                #'multinomial-naive-bayes__fit_prior': [True, False],
#                'multinomial-naive-bayes__class_prior': [None]
#            }

#SVC
#parameters = {  'svc__C': [1, 10, 100, 1000],
#                'svc__kernel': ['rbf'],
#                #'svc__degree': [3],
#                'svc__gamma': [1, 0.1],
#                #'svc__coef0': [0.0],
#                #'svc__shrinking': [True],
#                #'svc__probability': [False],
#                #'svc__tol': [1e-3],
#                #'svc__cache_size': [200],
#                #'svc__class_weight': [None],
#                #'svc__verbose': [False],
#                #'svc__max_iter': [-1],
#                #'svc__decision_function_shape': ['ovr', 'ovo'],
#                #'svc__break_ties': [False],
#                'svc__random_state': [1, None]
#            }

#RANDOMFOREST
#parameters = { #'randomforest__n_estimators': [100],
#               'randomforest__criterion': ['gini', 'entropy'],
#               'randomforest__max_depth': [9, 10, None],
#               'randomforest__min_samples_split': [9, 10, 11],
#               'randomforest__min_samples_leaf': [1, 2, 3],
#               #'randomforest__min_weight_fraction_leaf': [0.0],
#               'randomforest__max_features': [8, 9, 10],
#               #'randomforest__max_leaf_nodes': [None],
#               #'randomforest__min_impurity_decrease': [0.0],
#               #'random_forest__bootstrap': [True, False],
#               #'randomforest__oob_score': [False],
#               #'randomforest__n_jobs': [None],
#               'randomforest__random_state': [1, None],
#               #'randomforest__verbose': [0],
#               #'randomforest__warm_start': [False],
#               #'randomforest__class_weight': [None],
#               #'randomforest__ccp_alpha': [0.0],
#               #'randomforest__max_samples': [False],
#              }

#ADABOOST + DECISION TREE
#parameters = { #'adaboost__base_estimator__criterion': ['gini', 'entropy'],
#               #'adaboost__base_estimator__splitter': ['best', 'random'],
#               #'adaboost__base_estimator__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
#               #'adaboost__base_estimator__min_samples_split': [2, 3, 4, 5],
#               #'adaboost__base_estimator__min_samples_leaf': [1, 2, 3],
#               #'adaboost__base_estimator__random_state': [1, None],
#               'adaboost__n_estimators': [10, 50, 100, 500],
#               'adaboost__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
#               #'adaboost__algorithm': ['SAMME', 'SAMME.R'],
#               #'adaboost__random_state': [1, None]
#               }

##ADABOOST + MULTINOMIALNB
#parameters = { 'adaboost__base_estimator__alpha': [0.01, 0.1, 0.5],
#               'adaboost__base_estimator__fit_prior': [True, False],
#               #'adaboost__base_estimator__class_prior': [None],
#               'adaboost__n_estimators': [10, 50, 100, 500],
#               'adaboost__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
#               #'adaboost__algorithm': ['SAMME', 'SAMME.R'],
#               'adaboost__random_state': [1, None]
#               }

#ADABOOST + SVC
#parameters = { #'adaboost__base_estimator__C':[1, 10, 100],
#               #'adaboost__base_estimator__gamma':[1, 0.1, 0.01],
#               'adaboost__base_estimator__kernel':['linear'],
#               'adaboost__base_estimator__probability': [True],
#               'adaboost__n_estimators': [10, 50, 100, 500],
#               'adaboost__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
#               #'adaboost__algorithm': ['SAMME', 'SAMME.R'],
#               'adaboost__random_state': [1, None]
#               }

#ADABOOST + RANDOM FOREST
#parameters = { 'adaboost__base_estimator__criterion': ['gini', 'entropy'],
#               'adaboost__base_estimator__max_depth': [9, 10, None],
#               'adaboost__base_estimator__min_samples_split': [9, 10, 11, 12],
#               'adaboost__base_estimator__min_samples_leaf': [1, 2, 3],
#               'adaboost__base_estimator__max_features': [8, 9, 10],
#               'adaboost__base_estimator__random_state': [1, None],
#               'adaboost__n_estimators': [10, 50, 100, 500],
#               'adaboost__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
#               #'adaboost__algorithm': ['SAMME', 'SAMME.R'],
#               'adaboost__random_state': [1, None]
#               }

#XGBOOST
#parameters = {  'xgboost__nthread': [4], #when use hyperthread, xgboost may become slower
#                'xgboost__learning_rate': [0.01], #so called `eta` value
#                'xgboost__max_depth': [7],
#                #'xgboost__min_child_weight': [11],
#                'xgboost__subsample': [0.8],
#                'xgboost__colsample_bytree': [0.8],
#                'xgboost__n_estimators': [1000], #number of trees, change it to 1000 for better results
#                #'xgboost__missing': [-999],
#                #'xgboost__seed': [1337],
#                #'xgboost__booster': ['gbdt'],
#                #'xgboost__metric': ['multiclass'],
#                'xgboost__eval_metric': ['mlogloss'],
#                #'xgboost__silent': [False], 
#                #'xgboost__scale_pos_weight': [1],  
#                #'xgboost__subsample': [0.8],
#                'xgboost__objective': ['multi:softmax'], 
#                'xgboost__reg_alpha': [0.3],
#                'xgboost__gamma': [0, 1],
#                'xgboost__use_label_encoder': [False],
#                'xgboost__num_class': [3]
#            }

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
#results = cross_validate(model, X, y, cv=kfold)
#print("Average accuracy: %f (%f)" %(results['test_score'].mean(), results['test_score'].std()))
grid = GridSearchCV(model, param_grid=parameters, cv=kfold, n_jobs=-1)
grid.fit(X_train, y_train)

y_pred = grid.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print('Accuracy of the best classifier after CV is %.3f%%' % (accuracy*100))

print(classification_report(y_pred,y_test, target_names=['news', 'opinion', 'fake_news']))

#optmised_tree = grid.best_estimator_
#Melhor parâmetro
print ("Melhor parâmetro:", grid.best_params_)

#Gerar dataframe de resultados
result_df = pd.DataFrame(grid.cv_results_)

#DECISION TREE
#result_df = result_df.to_csv("tweet_text_stemmed_grid_decisiontree.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_decisiontree_smote.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_decisiontree_smote_randomundersampler.csv", sep=";", index=False)

#MULTINOMIALNB
#result_df = result_df.to_csv("tweet_text_stemmed_grid_multinomialnb.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_multinomialnb_smote.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_multinomialnb_smote_randomundersampler.csv", sep=";", index=False)

#SVC
#result_df = result_df.to_csv("tweet_text_stemmed_grid_svc.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_svc_smote.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_svc_smote_randomundersampler.csv", sep=";", index=False)

#RANDOM FOREST
#result_df = result_df.to_csv("tweet_text_stemmed_grid_randomforest.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_randomforest_smote.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_randomforest_smote_randomundersampler.csv", sep=";", index=False)

#ADABOOST + DECISION TREE
#result_df = result_df.to_csv("tweet_text_stemmed_grid_ababoost_decisiontree.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_ababoost_decisiontree_smote.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_ababoost_decisiontree_smote_randomundersampler.csv", sep=";", index=False)

#ADABOOST + MULTINOMIALNB
#result_df = result_df.to_csv("tweet_text_stemmed_grid_ababoost_multinomialnb.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_ababoost_multinomialnb_smote.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_ababoost_multinomialnb_smote_randomundersampler.csv", sep=";", index=False)

#ADABOOST + SVC
#result_df = result_df.to_csv("tweet_text_stemmed_grid_ababoost_svc.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_ababoost_svc_smote.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_ababoost_svc_smote_randomundersampler.csv", sep=";", index=False)

#ADABOOST + RANDOM FOREST
#result_df = result_df.to_csv("tweet_text_stemmed_grid_ababoost_randomforest.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_ababoost_randomforest_smote.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_ababoost_randomforest_smote_randomundersampler.csv", sep=";", index=False)

#XGBOOST
#result_df = result_df.to_csv("tweet_text_stemmed_grid_xgboost.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_xgboost_smote.csv", sep=";", index=False)
#result_df = result_df.to_csv("tweet_text_stemmed_grid_xgboost_smote_randomundersampler.csv", sep=";", index=False)

#print (result_df)

#print (grid.cv_results_)
#print (grid.best_estimator_)
#print (grid.scorer_)
#print (grid.n_splits_)

end = timeit.default_timer()
print ('Duração: %f' % (end - start))