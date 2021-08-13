# PIVIC: "Um modelo computacional para identificação de notícias falsas sobre a Covid-19 no Brasil"
# Code: Preprocessador de Texto contendo todas as estapas de pre-processamento utilizando Pandas.
# Author: Anísio Pereira Batista Filho

##Essentials
import os
import csv
import numpy as np ##Numpy
import pandas as pd ##Pandas
##Matplotlib
import matplotlib
import matplotlib.pyplot as plt
##Ekphrasis
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
##NLTK
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
##Cogroo4py
#from utils.cogroo4py.cogroo_interface import cogroo
##Wordcloud
from wordcloud import WordCloud, ImageColorGenerator
##Tweepy
import tweepy as tw
##Geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
##Utils
import re
import unicodedata
from datetime import datetime
from itertools import islice
from tqdm.auto import tqdm
import time
import timeit

start = timeit.default_timer()

##Gerar dataframe de amostras:
def df_get_samples(df, qnt):
    df_sample = df.sample(n=qnt)
    return df_sample

##Eliminar duplicatas em um dataframe:
def df_drop_duplicates(df):
    df = df.drop_duplicates(subset='tweet_id', keep='first')
    return df

##Comparador de treinamentos (compara dois treinamentos e cria um conjunto de iguais e um de diferentes)
def compare_trainings(dfA, dfB):
    listC = []
    listD = []
    for a in tqdm(dfA.itertuples(), total=dfA.shape[0]):
        for b in dfB.itertuples():
            if a.tweet_id == b.tweet_id:
                if a.label == b.label:
                    dictC = dict({'tweet_id':a.tweet_id,'created_at':a.created_at,'user_location':a.user_location,'tweet_text':a.tweet_text,'label_A':a.label,'label_B':b.label,'pct_certainty_A':a.pct_certainty,'pct_certainty_B':b.pct_certainty})
                    listC.append(dictC)
                else:
                    dictD = dict({'tweet_id':a.tweet_id,'created_at':a.created_at,'user_location':a.user_location,'tweet_text':a.tweet_text,'label_A':a.label,'label_B':b.label,'pct_certainty_A':a.pct_certainty,'pct_certainty_B':b.pct_certainty})
                    listD.append(dictD)     

    dfC = pd.DataFrame(listC, columns=['tweet_id','created_at','user_location','tweet_text','label_A','label_B','pct_certainty_A','pct_certainty_B'])
    dfD = pd.DataFrame(listD, columns=['tweet_id','created_at','user_location','tweet_text','label_A','label_B','pct_certainty_A','pct_certainty_B'])

    dfC.to_csv("data/data-twitter/training/rotulaçao[iguais]2.csv", sep=",", index=False)
    dfD.to_csv("data/data-twitter/training/rotulaçao[diferentes]2.csv", sep=",", index=False)

##Calculo de Cohen Kappa:
def cohen_kappa(df, dfA, dfB):
    ##Criando dicionário de rótuloxquantidade com as rotulações do avaliador A:
    labelerA = dict()
    fake_news = opinion = news = 0
    for row in dfA.itertuples():
        if row.label == 1:
            fake_news += 1
            labelerA['fake_news'] = fake_news
        else: 
            if row.label == 0:
                opinion += 1
                labelerA['opinion'] = opinion
            else:
                if row.label == -1:
                    news += 1
                    labelerA['news'] = news
    print('Rotulador A: ',labelerA)

    ##Calculando a probabilidade de concordancia randomica do avaliador A:
    peA = dict()
    peA['fake_news'] = labelerA['fake_news']/dfA.shape[0]
    peA['opinion'] = labelerA['opinion']/dfA.shape[0]
    peA['news'] = labelerA['news']/dfA.shape[0]
    print('peA: ',peA)

    ##Criando dicionário de rótuloxquantidade com as rotulações do avaliador B:
    labelerB = dict()
    fake_news = opinion = news = 0
    for row in dfB.itertuples():
        if row.label == 1:
            fake_news += 1
            labelerB['fake_news'] = fake_news
        else: 
            if row.label == 0:
                opinion += 1
                labelerB['opinion'] = opinion
            else:
                if row.label == -1:
                    news += 1
                    labelerB['news'] = news
    print('Rotulador B: ',labelerB)

    ##Calculando a probabilidade de concordancia randomica do avaliador B:
    peB = dict()
    peB['fake_news'] = labelerB['fake_news']/dfB.shape[0]
    peB['opinion'] = labelerB['opinion']/dfB.shape[0]
    peB['news'] = labelerB['news']/dfB.shape[0]
    print('peB: ',peB)

    ##Calculando a probabilidade de ambos os avaliadores (para fake_news, opinion e news):
    pe = dict()
    pe['fake_news'] = peA['fake_news']*peB['fake_news']
    pe['opinion'] = peA['opinion']*peB['opinion']
    pe['news'] = peA['news']*peB['news']
    print('pe:',pe)

    ##Calculando a probabilidade de aceitação total:
    Pr = pe['fake_news']+pe['opinion']+pe['news']
    print('Pr: ',Pr)

    ##Calculando Po
    Po = df.shape[0]/dfA.shape[0]
    print('Po: ',Po)
    ##Calculando Kappa:
    K = (Po - Pr)/(1 - Pr)
    print('K: ',K)

def datetime_interval(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    dtidict = dict()
    dtidict['Ínicio do intervalo'] = df['created_at'].min()
    dtidict['Fim do invervalo'] = df['created_at'].max()
    print(dtidict)

def create_new_labelbase(df_original, df_base4label, total):
    #Criar uma lista com os ids utilizados na base anterior
    used_tweetid_list = []
    for row in df_base4label.itertuples():
        used_tweetid_list.append(row.tweet_id)
    #print(used_tweetid_list)
    
    #Calcular o tamanho da nova base desejada
    qnt = total - df_base4label.shape[0]
    #print(qnt)
    #Criar o novo dataframe
    df_newbase4label = pd.DataFrame()
    #Atribuir as amostras (samples) ao novo dataframe
    df_newbase4label = df_original.sample(n=qnt)
    #df_newbase4label = df_newbase4label.set_index("tweet_id")
    print('Tamanho da base após pegar as primeiras samples: ', df_newbase4label.shape[0])

    flag = 0
    check = 0
    while (flag != 1):
        flag = 0
        check = 0
        print('-Começo de novo loop-')
        #Verificar se existem duplicatas nas amostras, caso tenha elimina e pega novas amostras
        #O loop termina quando não existem mais duplicatas no dataframe
        while(check != 1):
            print('-Dentro do while-')
            #print('-Entrou no loop interno-')
            df_newbase4label = df_newbase4label.drop_duplicates(subset='tweet_id', keep='first')
            #df_newbase4label.reset_index().drop_duplicates(subset='tweet_id', keep='first').set_index('tweet_id')
            #df_newbase4label = df_newbase4label[~df_newbase4label.index.duplicated(keep='first')]
            print('Tamanho da base após dropar duplicatas: ', df_newbase4label.shape[0])
            undropped = df_newbase4label.shape[0]
            if undropped == qnt:
                check = 1
            else:
                df_newbase4label = df_newbase4label.append(df_original.sample(n=(qnt-undropped)), ignore_index=True)
                print('Tamanho da base após pegar novas samples após drop de duplicatas: ', df_newbase4label.shape[0])
        print('-Saiu do while-')
        print('-Começo do for-')
        
        #Com a lista de ids usados na primeira base para rotulação, ele verifica se no dataframe atual
        #existe alguma repetição desse mesmo id, caso tenha, o id repetido é eliminado do novo dataframe
        cont = 0
        for ind in used_tweetid_list:
            for row in df_newbase4label.itertuples():
                if ind == row.tweet_id:
                    #df_newbase4label = df_newbase4label.drop(ind, inplace=True)
                #df_newbase4label = df_newbase4label.drop(df_newbase4label.loc[df_newbase4label['tweet_id']==ind].index, inplace=True)
                    df_newbase4label = df_newbase4label.drop(df_newbase4label.index[df_newbase4label['tweet_id'] == ind])
                    cont += 1
        print('-Saiu do for-')
        print('Contador: ', cont)
        #Verificar tamanho da base após dropar duplicatas em relação a base original
        print('Tamanho da base depois tirar as cópias da base antiga: ', df_newbase4label.shape[0])
        #Caso o tamanho da base atual após os processos acimas sejam igual o tamanho desejado, 
        #ativa-se a flag para parar o loop principal
        if df_newbase4label.shape[0] == qnt:
            flag = 1
        print('Flag: ', flag)
        #df_newbase4label = df_newbase4label.set_index("tweet_id")
        df_newbase4label = df_newbase4label.append(df_original.sample(n=cont))
        print('Tamanho da base após pegar novas samples: ', df_newbase4label.shape[0])
    print('-Saiu do loop-')
    print ('Tamanho da base final após sair do while: ', df_newbase4label.shape[0])
    #print(df_newbase4label)
    return df_newbase4label

##Função main:
###Abertura de arquivo e criação do dataframe:
df_file = pd.read_csv("data/data-twitter/data-twitter-modfied-utf8.csv", sep=";")
df_original = pd.DataFrame(df_file)
df_original = df_original.drop_duplicates(subset='tweet_id', keep='first')

df_base4label_file = pd.read_csv("data/data-twitter/base_para_rotulaçao.csv", sep=",")
df_base4label = pd.DataFrame(df_base4label_file)
df_base4label = df_base4label.drop_duplicates(subset='tweet_id', keep='first')

df_labeliguais_file = pd.read_csv("data/data-twitter/training/rotulaçao[iguais]_complete.csv", sep=";")
df_labeliguais = pd.DataFrame(df_labeliguais_file)
df_labeliguais = df_labeliguais.drop_duplicates(subset='tweet_id', keep='first')

#df_newbase4label_file = pd.read_csv("data/data-twitter/base_para_rotulaçao2.csv", sep=",")
#df_newbase4label = pd.DataFrame(df_newbase4label_file)
#df_newbase4label = df_base4label.drop_duplicates(subset='tweet_id', keep='first')

#df_file = pd.read_csv("data/data-twitter/data-twitter_upgraded.csv", sep=";")
#df = pd.DataFrame(df_file)
#df = df.drop_duplicates(subset='tweet_id', keep='first')

#df_file = pd.read_csv("data/data-twitter/training/rotulaçao[iguais]_complete.csv", sep=";")
#df = pd.DataFrame(df_file)
#df = df.drop_duplicates(subset='tweet_id', keep='first')

#dfA_file = pd.read_csv("data/data-twitter/training/rotulaçao[anisiofilho].csv", sep=",")
#dfA = pd.DataFrame(dfA_file)
#dfA = dfA.drop_duplicates(subset='tweet_id', keep='first')

#dfB_file = pd.read_csv("data/data-twitter/training/rotulaçao[debora].csv", sep=",")
#dfB = pd.DataFrame(dfB_file)
#dfB = dfB.drop_duplicates(subset='tweet_id', keep='first')

#dfI_file = pd.read_csv("data/data-twitter/training/rotulaçao[iguais].csv", sep=",")
#dfI = pd.DataFrame(dfI_file)

#dfD_file = pd.read_csv("data/data-twitter/training/rotulaçao[diferentes].csv", sep=",")
#dfD = pd.DataFrame(dfD_file)

#dfI2_file = pd.read_csv("data/data-twitter/training/rotulaçao[iguais]2.csv", sep=",")
#dfI2 = pd.DataFrame(dfI2_file)

#dfD2_file = pd.read_csv("data/data-twitter/training/rotulaçao[diferentes]2.csv", sep=",")
#dfD2 = pd.DataFrame(dfD2_file)

#df_file = pd.read_csv("data/data-twitter/training/rotulaçao[iguais].csv", sep=',')
#df = pd.DataFrame(df_file)

###Chamadas de funções:
datetime_interval(df_original)
print('Tamanho da base original:', df_original.shape[0])
print('Tamanho da base rotulada:', df_base4label.shape[0])
print('Tamanho da base rotulada igual:', df_labeliguais.shape[0])
#compare_trainings(dfA,dfB)
#cohen_kappa(df, dfA, dfB)
#print(dfI.shape[0])
#print(dfD.shape[0])
#print(dfI2.shape[0])
#print(dfD2.shape[0])
#dfnewbase4label = create_new_labelbase(df_original, df_base4label, 5000)

###Salvando alterações no csv:
#add_file = df.to_csv("data/data-twitter/training/rotulaçao[iguais]_complete2.csv", sep=";", index=False)
#add_file = dfnewbase4label.to_csv("data/data-twitter/base_para_rotulaçao2.csv", sep=",", index=False)

end = timeit.default_timer()
print ('Duração: %f' % (end - start))