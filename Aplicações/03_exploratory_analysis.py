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
##Tweepy
import tweepy as tw
##Geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
##Spacy
import spacy
##Wordcloud
from wordcloud import WordCloud, ImageColorGenerator
##Utils
import re
import unicodedata
from datetime import datetime
from itertools import islice
from tqdm.auto import tqdm
import time
import timeit

start = timeit.default_timer()

##Listador de turnos:
def timeshift_listing(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    time_shift = []
    for row in tqdm(df.itertuples(), total=df.shape[0]): 
        line = row.created_at.strftime("%H:%M:%S")
        if (line > '00:00:00' and line < '11:59:59'):
            time_shift.append('manhã')
        else:
            if (line > '12:00:00' and line < '17:59:59'):
                time_shift.append('tarde')
            else: 
                time_shift.append('noite')
    
    try:
        df.insert(2,'time_shift', time_shift)
    except:
        df['time_shift'] = time_shift

##Crawler de usuários:
def user_crawler(df):
    ###Twitter Developer Keys:
    # Twitter Developer Keys for crawler_pivic app
    consumer_key = '<consumer_key>'
    consumer_secret = '<consumer_secret>'
    access_token = '<access_token>'
    access_token_secret = '<access_token_secret>'

    ###Autgenticação entre Twitter Developer e este script:
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    ###Ínicio da recuperação de usuários:
    sleepTime = 2
    id_list = []
    user_screen_name = []
    user_id = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        id_list.append(row.tweet_id)
    for tweet_id in tqdm(id_list, total=len(id_list)):
        try:
            tweetFetched = api.get_status(tweet_id)
            #Para saber o que poder ser recuperado pelo objeto status ou objeto user, pesquisar.
            #print("User screen name: " + tweetFetched.user.screen_name)
            #print("User id: " + tweetFetched.user.id_str)
            #print("User name: " + tweetFetched.user.name)        
            user_screen_name.append(tweetFetched.user.screen_name)
            user_id.append(tweetFetched.user.id_str)
            #user_name.append(tweetFetched.user.name)
            #trainingDataSet.append(tweetFetched)
            time.sleep(sleepTime)        
        except:
            user_screen_name.append('invalid_user')
            user_id.append('invalid_user')
            #user_name.append('invalid_user')
            #print("Inside the exception - no:2")
            continue
        
    try:
        df.insert(3,'user_screen_name', user_screen_name)
    except:
        df['user_screen_name'] = user_screen_name
    try:
        df.insert(4,'user_id', user_id)
    except:
        df['user_id'] = user_id

##Tratamento da coluna de localização:
def location_trater(df):
    abreviações_estados_file = pd.read_csv("data/utils/abreviações_estados.csv", sep=",")
    df_estados = pd.DataFrame(abreviações_estados_file)
    df['user_location'] = df['user_location'].fillna("invalidlocation")
    local = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        line = row.user_location
        line = line.lower()
        nfkd = unicodedata.normalize('NFKD', line)
        new_line = u"".join([c for c in nfkd if not unicodedata.combining(c)])
        symboless_line = re.sub('[^a-zA-Z0-9 \\\\]', ' ', new_line)
        symboless_line = " ".join(symboless_line.split())
        local.append(symboless_line)

    ab_estados = dict()
    for row in tqdm(df_estados.itertuples(), total=df_estados.shape[0]):
        ab_estados[row.sigla] = row.estado
    #print (ab_estados)
    stateLocal = []
    for l in tqdm(local, total=len(local)):
        for s in l.split():
            if s in ab_estados:
                #print (l.split())
                l = l.replace(s, ab_estados[s])
            #print (l)
        #l = " ".join(re.split(r"\s+", l))
        #print (l)
        #if l == "invalid location":
            #l = "invalidlocation"
        stateLocal.append(l)
        #print (stateLocal)
    
    try:
        df.insert(6,'location_treated', stateLocal)
    except:
        df['location_treated'] = stateLocal

##Gerador de listas de estados e regiçoes utilizando Geopy:
def geopy_stateregion(df):
    geolocator = Nominatim(user_agent="google")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    states_regions_br_file = pd.read_csv("data/utils/estados_abreviações_regiões.csv", sep=",")
    df_states_regions_br = pd.DataFrame(states_regions_br_file)
    states_regions_br = dict()
    for row in tqdm(df_states_regions_br.itertuples(), total=df_states_regions_br.shape[0]):
        states_regions_br[row.estado] = row.região

    states = []
    regions = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        try:
            local = geolocator.geocode(row.location_treated, addressdetails=True)
            #location.append(geolocator.geocode(row['user_localization'], addressdetails=True))
            local = local.raw
            local = local['address']
            #local_file.write("%s\n" % line)
            if local['country_code'] == 'br':
                try:
                    if local['state'] in states_regions_br:
                        states.append(local['state'])
                        regions.append(states_regions_br[local['state']])
                except:
                    states.append('undefinedstate')
                    regions.append('undefinedregion')
            else:
                states.append('notbrazilstate')
                #regions.append(local['country'])#pegar o país//anterior: 'notbrazilregion'
                regions.append('notbrazilregion')
        except:
            local = {'state': 'invalid', 'region': 'invalid', 'country': 'invalid'}
            states.append('invalidstate')
            regions.append('invalidregion')

    try:
        df.insert(7,'state_location', states)
    except:
        df['state_location'] = states
    try:
        df.insert(8,'region_location', regions)
    except:
        df['region_location'] = regions

##Contador de caracteres:
def tweet_charcounter(df, rowname):
    charcountlist = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        line = str(row[rowname])
        charcountlist.append(len(line))
    
    return charcountlist

##Contador de palavras:
def tweet_wordcounter(df, rowname):
    wordcountlist = []
    for index, row in tqdm(df.iterrows(),total=df.shape[0]):
        wordstring = str(row[rowname])
        wordlist = wordstring.split()
        wordcountlist.append(len(wordlist))

    return wordcountlist

##Gerador de lista de turnos:
def timeshift_list_generator(df):
    timeshift_list = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        timeshift_list.append(row.time_shift)
    return timeshift_list

##Gerador de lista de nome de usuários:
def user_screen_name_list_generator(df):
    userlist = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        userlist.append(row.user_screen_name)
    return userlist

##Gerador de lista de estados:
def state_list_generator(df):
    statelist = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        statelist.append(row.state_location)
    return statelist

##Gerador de lista de regiões:
def region_list_generator(df):
    regionlist = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        regionlist.append(row.region_location)
    return regionlist

##Criando os dicionários de Fake News e News separando por turno de horário:
def labelpertimeshift(df):
    fakenewsdict = dict()
    newsdict = dict()
    opiniondict = dict()
    manhãfn = tardefn = noitefn = manhãn = tarden = noiten = manhão = tardeo = noiteo = 0
    for row in tqdm(df.itertuples(),total=df.shape[0]):
        if row.label_A == 1: #Poderia ser label_B também
            if row.time_shift == 'manhã':
                manhãfn += 1
                fakenewsdict['Manhã'] = manhãfn
            elif row.time_shift == 'tarde':
                tardefn += 1
                fakenewsdict['Tarde'] = tardefn
            elif row.time_shift == 'noite':
                noitefn += 1
                fakenewsdict['Noite'] = noitefn
        elif row.label_A == -1: ##Poderia ser label_B também
            if row.time_shift == 'manhã':
                manhãn += 1
                newsdict['Manhã'] = manhãn
            elif row.time_shift == 'tarde':
                tarden += 1
                newsdict['Tarde'] = tarden
            elif row.time_shift == 'noite':
                noiten += 1
                newsdict['Noite'] = noiten
        elif row.label_A == 0: ##Poderia ser label_B também
            if row.time_shift == 'manhã':
                manhão += 1
                opiniondict['Manhã'] = manhão
            elif row.time_shift == 'tarde':
                tardeo += 1
                opiniondict['Tarde'] = tardeo
            elif row.time_shift == 'noite':
                noiteo += 1
                opiniondict['Noite'] = noiteo

    fakenewsdict = dict(sorted(fakenewsdict.items(), key=lambda x: x[0].lower()))
    newsdict = dict(sorted(newsdict.items(), key=lambda x: x[0].lower()))
    opiniondict = dict(sorted(opiniondict.items(), key=lambda x: x[0].lower()))
    labelpertimeshiftdict = dict()
    labelpertimeshiftdict['Fake News'] = fakenewsdict
    labelpertimeshiftdict['News'] = newsdict
    labelpertimeshiftdict['Opinion'] = opiniondict
    #print('Fake News Dict: ',fakenewsdict)
    #print('News Dict: ',newsdict)
    #print('Label per Time Shift Dict: ',labelpertimeshiftdict)
    
    return labelpertimeshiftdict

##Criando dicionário de Fake News e News e separando por regiões:
def labelperregion(df):
    ##Criando os dicionários de Fake News e News separando por região:
    fakenewsdict = dict()
    newsdict = dict()
    opiniondict = dict()
    nortefn = nordestefn = sulfn = sudestefn = centro_oestefn = undefinedregionfn = notbrazilregionfn = invalidregionfn = 0
    norten = nordesten = suln = sudesten = centro_oesten = undefinedregionn = notbrazilregionn = invalidregionn = 0
    norteo = nordesteo = sulo = sudesteo = centro_oesteo = undefinedregiono = notbrazilregiono = invalidregiono = 0
    for row in tqdm(df.itertuples(),total=df.shape[0]):
        if row.label_A == 1: #Poderia ser label_B também
            if row.region_location == 'Norte':
                nortefn += 1
                fakenewsdict['Norte'] = nortefn
            elif row.region_location == 'Nordeste':
                nordestefn += 1
                fakenewsdict['Nordeste'] = nordestefn
            elif row.region_location == 'Sul':
                sulfn += 1
                fakenewsdict['Sul'] = sulfn
            elif row.region_location == 'Sudeste':
                sudestefn += 1
                fakenewsdict['Sudeste'] = sudestefn
            elif row.region_location == 'Centro-Oeste':
                centro_oestefn += 1
                fakenewsdict['Centro-Oeste'] = centro_oestefn
            elif row.region_location == 'undefinedregion':
                undefinedregionfn += 1
                fakenewsdict['undefinedregion'] = undefinedregionfn
            elif row.region_location == 'notbrazilregion':
                notbrazilregionfn += 1
                fakenewsdict['notbrazilregion'] = notbrazilregionfn
            elif row.region_location == 'invalidregion':
                invalidregionfn += 1
                fakenewsdict['invalidregion'] = invalidregionfn
        if row.label_A == -1: ##Poderia ser label_B também
            if row.region_location == 'Norte':
                norten += 1
                newsdict['Norte'] = norten
            elif row.region_location == 'Nordeste':
                nordesten += 1
                newsdict['Nordeste'] = nordesten
            elif row.region_location == 'Sul':
                suln += 1
                newsdict['Sul'] = suln
            elif row.region_location == 'Sudeste':
                sudesten += 1
                newsdict['Sudeste'] = sudesten
            elif row.region_location == 'Centro-Oeste':
                centro_oesten += 1
                newsdict['Centro-Oeste'] = centro_oesten
            elif row.region_location == 'undefinedregion':
                undefinedregionn += 1
                newsdict['undefinedregion'] = undefinedregionn
            elif row.region_location == 'notbrazilregion':
                notbrazilregionn += 1
                newsdict['notbrazilregion'] = notbrazilregionn
            elif row.region_location == 'invalidregion':
                invalidregionn += 1
                newsdict['invalidregion'] = invalidregionn
        if row.label_A == 0: ##Poderia ser label_B também
            if row.region_location == 'Norte':
                norteo += 1
                opiniondict['Norte'] = norteo
            elif row.region_location == 'Nordeste':
                nordesteo += 1
                opiniondict['Nordeste'] = nordesteo
            elif row.region_location == 'Sul':
                sulo += 1
                opiniondict['Sul'] = sulo
            elif row.region_location == 'Sudeste':
                sudesteo += 1
                opiniondict['Sudeste'] = sudesteo
            elif row.region_location == 'Centro-Oeste':
                centro_oesteo += 1
                opiniondict['Centro-Oeste'] = centro_oesteo
            elif row.region_location == 'undefinedregion':
                undefinedregiono += 1
                opiniondict['undefinedregion'] = undefinedregiono
            elif row.region_location == 'notbrazilregion':
                notbrazilregiono += 1
                opiniondict['notbrazilregion'] = notbrazilregiono
            elif row.region_location == 'invalidregion':
                invalidregiono += 1
                opiniondict['invalidregion'] = invalidregiono

    fakenewsdict = dict(sorted(fakenewsdict.items(), key=lambda x: x[0].lower()))
    newsdict = dict(sorted(newsdict.items(), key=lambda x: x[0].lower()))
    opiniondict = dict(sorted(opiniondict.items(), key=lambda x: x[0].lower()))
    labelperregiondict = dict()
    labelperregiondict['Fake News'] = fakenewsdict
    labelperregiondict['News'] = newsdict
    labelperregiondict['Opinion'] = opiniondict
    #print('Fake News Dict: ',fakenewsdict)
    #print('News Dict: ',newsdict)
    #print('Opinion Dict',opiniondict)
    #print('Label per Region Dict: ',labelperregiondict)
    
    return labelperregiondict

##Criando dicionário de Fake News e News e separando por Top 10 nome de usuário:
def labelperuserscreenname(df):
    ##Criando lista de frequência dos usuários
    userlist = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        userlist.append(row.user_screen_name)

    userfreq = []
    userfreq = [userlist.count(p) for p in tqdm(userlist, total=len(userlist))]
    userfreqdict = dict(list(zip(userlist,userfreq)))
    userfreqdict = dict(sorted(userfreqdict.items(), key=lambda t: t[1], reverse=True))

    ##Criando os dicionários de Fake News, News e Opinion separando por nome de usuário:
    usrdict = dict()
    for usr in islice(tqdm(userfreqdict, total=10),0,10):
        fake_news = news = opinion = 0
        supportdict = dict()
        for row in df.itertuples():
            if usr == row.user_screen_name:
                if row.label_A == 1: #Poderia ser label_B também
                    fake_news += 1
                    supportdict['Fake News'] = fake_news
                else:
                    supportdict['Fake News'] = fake_news
                if row.label_A == -1: ##Poderia ser label_B também
                    news += 1
                    supportdict['News'] = news
                else: 
                    supportdict['News'] = news
                if row.label_A == 0: ##Poderia ser label_B também
                    opinion += 1
                    supportdict['Opinion'] = opinion
                else: 
                    supportdict['Opinion'] = opinion
    
        usrdict[usr] = supportdict
    
    return usrdict

##Distribuição de tuítes por classes:
def class_distribution(dfA, dfB):
    ##Criando dicionário de rótuloxquantidade com as rotulações do avaliador A:
    labelerA = dict()
    fake_news = opinion = news = 0
    for row in tqdm(dfA.itertuples(),total=dfA.shape[0]):
        if row.label == 1:
            fake_news += 1
            labelerA['Fake News'] = fake_news
        else: 
            if row.label == 0:
                opinion += 1
                labelerA['Opinion'] = opinion
            else:
                if row.label == -1:
                    news += 1
                    labelerA['News'] = news

    ##Criando dicionário de rótuloxquantidade com as rotulações do avaliador B:
    labelerB = dict()
    fake_news = opinion = news = 0
    for row in tqdm(dfB.itertuples(), total=dfB.shape[0]):
        if row.label == 1:
            fake_news += 1
            labelerB['Fake News'] = fake_news
        else: 
            if row.label == 0:
                opinion += 1
                labelerB['Opinion'] = opinion
            else:
                if row.label == -1:
                    news += 1
                    labelerB['News'] = news

    labelerA = dict(sorted(labelerA.items(), key=lambda x: x[0].lower()))
    labelerB = dict(sorted(labelerB.items(), key=lambda x: x[0].lower()))
    print('Rotulador A: ',labelerA)
    print('Rotulador B: ',labelerB)
    labeldict = dict()
    labeldict['Rotulador A'] = labelerA
    labeldict['Rotulador B'] = labelerB

    return labeldict

##Análise morfológica:
def morphologic_analysis(df, rowname):
    print('-Início do morphologic_analysis-')
    token_list = []
    nlp = spacy.load('pt_core_news_lg')
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        doc = nlp(str(row[rowname]))
        doc_list = []
        for token in doc:
            doc_list.append(token.pos_)
            #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                    #token.shape_, token.is_alpha, token.is_stop)
        token_list.append(doc_list)
    #print(token_list)
    return token_list

##Dicionário baseado nas palavras NAVA
def navadictmaker(token_list):
    print('-Início do navadictmaker-')
    substlist = []
    adjlist = []
    verblist = []
    advlist = []
    for internlist in tqdm(token_list, total=len(token_list)):
        substantivos = adjetivos = verbos = adverbios = 0
        for ind in internlist:
            if ind == 'NOUN':
                substantivos += 1
            if ind == 'ADJ':
                adjetivos += 1
            if ind == 'VERB':
                verbos += 1
            if ind == 'ADV':
                adverbios += 1
        substlist.append(substantivos)
        adjlist.append(adjetivos)
        verblist.append(verbos)
        advlist.append(adverbios)
    navadict = dict()
    navadict['Substantivos'] = substlist
    navadict['Adjetivos'] = adjlist
    navadict['Verbos'] = verblist
    navadict['Adverbios'] = advlist
    
    return navadict

##Gerador de dicionário de frequências:
def frequency_generator(word_list = []):
    #wordstring = " ".join(word_list)
    #wordlist = wordstring.split()
    wordlist = word_list
    wordfreq = []
    wordfreq = [wordlist.count(p) for p in tqdm(wordlist, total=len(wordlist))]
    freqdict = dict(list(zip(wordlist,wordfreq)))
    sorteddict = dict(sorted(freqdict.items(), key=lambda t: t[1], reverse=True))

    return sorteddict

##Gerador de dicionário de TOP 10 de frequências:
def frequency_generator_top10(word_list = []):
    wordstring = " ".join(word_list)
    wordlist = wordstring.split()
    wordfreq = []
    wordfreq = [wordlist.count(p) for p in tqdm(wordlist, total=len(wordlist))]
    freqdict = dict(list(zip(wordlist,wordfreq)))
    sorteddict = dict(sorted(freqdict.items(), key=lambda t: t[1], reverse=True))

    freqdicttop10 = dict()
    for k in islice(tqdm(sorteddict, total=10),0,10):
        freqdicttop10[k] = sorteddict[k] 

    return freqdicttop10
 
##Contador de frequência de palavras:
def wordfrequency(word_list = []):
    wordstring = " ".join([str(i) for i in word_list])
    wordlist = wordstring.split()
    wordfreq = []
    wordfreq = [wordlist.count(p) for p in tqdm(wordlist, total=len(wordlist))]
    freqdict = dict(list(zip(wordlist,wordfreq)))
    sorteddict = dict(sorted(freqdict.items(), key=lambda t: t[1], reverse=True))

    frequency_list = []
    for s in tqdm(sorteddict.items(), total=len(sorteddict)): 
        #print(str(s))
        frequency_list.append(str(s))

    return frequency_list

##Gerador de lista de palavras:
def wordlist_generator(df, rowname):
    wordlists = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        wordstring = str(row[rowname])
        wordlist = wordstring.split()
        wordlists.append(wordlist)
    
    return wordlists

##Gerador de lista de palavras por classe:
def wordlistperlabel_generator(df, rowname, label):
    wordlists = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row['label_A'] == label:
            wordstring = str(row[rowname])
            wordlist = wordstring.split()
            wordlists.append(wordlist)
    
    return wordlists

##Gráfico de barras verticais simples:
def bar_simple(freqdict, xlabel, ylabel, title):
    description = list(freqdict.keys())
    values = list(freqdict.values())
    
    ###Início do código de barras simples:
    x = np.arange(len(description))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(16.00,9.00))
    rects = ax.bar(x, values, width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(description)
    #ax.legend()

    autolabel(rects, ax)
    fig.tight_layout()
    return plt

##Gráfico de barras verticais duplas:
def bar_double(labeldict, xlabel, ylabel, title):
    ##Criando o grafico com os valores dos dois dicionários:
    description = list(list(labeldict.values())[0].keys())
    valuesA = list(list(labeldict.values())[0].values())
    valuesB = list(list(labeldict.values())[1].values())

    x = np.arange(len(description))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(16.00,9.00))
    rects1 = ax.bar(x - width/2, valuesA, width, label=list(labeldict.keys())[0], color='tab:orange')
    rects2 = ax.bar(x + width/2, valuesB, width, label=list(labeldict.keys())[1], color='tab:blue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(description)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    fig.tight_layout()
    return plt

##Gráfico de barras verticais triplas para muitas descrições:
def bar_triplep(labeldict, xlabel, ylabel, title):
    ##Criando o grafico com os valores dos dois dicionários:
    description = list(list(labeldict.values())[0].keys())
    valuesA = list(list(labeldict.values())[0].values())
    valuesB = list(list(labeldict.values())[1].values())
    valuesC = list(list(labeldict.values())[2].values())

    x = np.arange(len(description))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(16.00,9.00))
    rects1 = ax.bar(x - width, valuesA, width, label=list(labeldict.keys())[0], color='tab:orange')
    rects2 = ax.bar(x, valuesB, width, label=list(labeldict.keys())[1], color='tab:blue')
    rects3 = ax.bar(x + width, valuesC, width, label=list(labeldict.keys())[2], color='tab:green')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(description)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)

    fig.tight_layout()
    return plt

##Gráfico de barras verticais triplas:
def bar_triple(labeldict, xlabel, ylabel, title):
    ##Criando o grafico com os valores dos três dicionários:
    #print (list(labeldict.keys()))
    #print (list(labeldict.values()))
    #print (list(list(labeldict.values())[0].keys())[0])
    #print (list(list(labeldict.values())[0].keys())[1])
    #print (list(list(labeldict.values())[0].keys())[2])
    #print (list(list(labeldict.values())[0].values())[2])

    description = list(labeldict.keys())
    valuesA = []
    valuesB = []
    valuesC = []
    for a in range(len(labeldict)):
        valuesA.append(list(list(labeldict.values())[a].values())[0])
    for b in range(len(labeldict)):
        valuesB.append(list(list(labeldict.values())[b].values())[1])
    for c in range(len(labeldict)):
        valuesC.append(list(list(labeldict.values())[c].values())[2])
    #print (valuesA)
    #print (valuesB)
    #print (valuesC)

    x = np.arange(len(description))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(16.00,9.00))
    rects1 = ax.bar(x - width, valuesA, width, label=(list(list(labeldict.values())[0].keys())[0]), color='tab:orange')
    rects2 = ax.bar(x, valuesB, width, label=(list(list(labeldict.values())[0].keys())[1]), color='tab:blue')
    rects3 = ax.bar(x + width, valuesC, width, label=(list(list(labeldict.values())[0].keys())[2]), color='tab:green')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(description)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)

    fig.tight_layout()
    return plt

##Gráfico de barras horizontais simples:
def barh_simple(freqdict, ylabel, xlabel, title):
    description = list(freqdict.keys())
    values = list(freqdict.values())
    
    ###Início do código de barras simples:
    y = np.arange(len(description))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(16.00,9.00))
    rects = ax.barh(y, values, width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_yticks(y)
    ax.set_yticklabels(description)
    #ax.legend()

    autolabelh(rects, ax)

    fig.tight_layout()
    return plt

##Autolabel para gráficos de barras verticias:
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    return

##Autolabel para gráficos de barras horizontais:
def autolabelh(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        width = rect.get_width()
        ax.annotate('{}'.format(width),
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(10, 0), 
                    textcoords="offset points",
                    ha='center', va='center')
    return

##Calcular a média de valores em uma coluna criando um diciónario por classes:
def calculate_mean(df, rowname):
    fakenewslist = []
    newslist = []
    opinionlist = []
    meandict = dict()

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row['label_A'] == -1:
            fakenewslist.append(row[rowname])
        elif row['label_A'] == 1:
            newslist.append(row[rowname])
        elif row['label_A'] == 0:
            opinionlist.append(row[rowname])
    
    meandict['Fake News'] = np.mean(fakenewslist)
    meandict['News'] = np.mean(newslist)
    meandict['Opinion'] = np.mean(opinionlist)
    
    return meandict

##Calcular a desvio padrão de valores em uma coluna criando um diciónario por classes:
def calculate_std(df, rowname):
    fakenewslist = []
    newslist = []
    opinionlist = []
    stddict = dict()

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row['label_A'] == -1:
            fakenewslist.append(row[rowname])
        elif row['label_A'] == 1:
            newslist.append(row[rowname])
        elif row['label_A'] == 0:
            opinionlist.append(row[rowname])
    
    stddict['Fake News'] = np.std(fakenewslist)
    stddict['News'] = np.std(newslist)
    stddict['Opinion'] = np.std(opinionlist)
    
    return stddict

##Criação de wordcloud:
def wordcloud_generator(word_list = []):
    all_summary = " ".join(word_list)
    wordcloud = WordCloud(
                      background_color='black', 
                      width=1600,                            
                      height=800).generate(all_summary)
    fig, ax = plt.subplots(figsize=(16,8))
    ax.imshow(wordcloud, interpolation='bilinear')       
    ax.set_axis_off()
    plt.imshow(wordcloud) 
    
    return wordcloud                
    #wordcloud.to_file('testcloud.png')

##Criação de wordclouds por rótulo
def wordcloud_wordlistperlabel_creator(df):
    ###Criação de wordclouds:
    ###Tweet_text_lower:
    ####Fake News:
    wlttlwfn = wordlistperlabel_generator(df, 'tweet_text_lower', -1)
    wfttlwfn = wordfrequency(wlttlwfn)
    wcttlwfn = wordcloud_generator(wfttlwfn)
    wcttlwfn.to_file("wordcloud/data-twitter/training/wordcloud_tweet_text_lower_fakenews-01.png")

    ####News:
    wlttln = wordlistperlabel_generator(df, 'tweet_text_lower', 1)
    wfttln = wordfrequency(wlttln)
    wcttln = wordcloud_generator(wfttln)
    wcttln.to_file("wordcloud/data-twitter/training/wordcloud_tweet_text_lower_news-01.png")

    ####Opinion:
    wlttlo = wordlistperlabel_generator(df, 'tweet_text_lower', 0)
    wfttlo = wordfrequency(wlttlo)
    wcttlo = wordcloud_generator(wfttlo)
    wcttlo.to_file("wordcloud/data-twitter/training/wordcloud_tweet_text_lower_opinion-01.png")

    ###Tweet_text_stemmed:
    ####Fake News:
    wlttstfn = wordlistperlabel_generator(df, 'tweet_text_stemmed', -1)
    wfttstfn = wordfrequency(wlttstfn)
    wcttstfn = wordcloud_generator(wfttstfn)
    wcttstfn.to_file("wordcloud/data-twitter/training/wordcloud_tweet_text_stemmed_fakenews-01.png")

    ####News:
    wlttstn = wordlistperlabel_generator(df, 'tweet_text_stemmed', 1)
    wfttstn = wordfrequency(wlttstn)
    wcttstn = wordcloud_generator(wfttstn)
    wcttstn.to_file("wordcloud/data-twitter/training/wordcloud_tweet_text_stemmed_news-01.png")

    ####Opinion:
    wlttsto = wordlistperlabel_generator(df, 'tweet_text_stemmed', 0)
    wfttsto = wordfrequency(wlttsto)
    wcttsto = wordcloud_generator(wfttsto)
    wcttsto.to_file("wordcloud/data-twitter/training/wordcloud_tweet_text_stemmed_opinion-01.png")

    ###Tweet_text_lemmatized:
    ####Fake News:
    wlttlmfn = wordlistperlabel_generator(df, 'tweet_text_lemmatized', -1)
    wfttlmfn = wordfrequency(wlttlmfn)
    wcttlmfn = wordcloud_generator(wfttlmfn)
    wcttlmfn.to_file("wordcloud/data-twitter/training/wordcloud_tweet_text_lemmatized_fakenews-01.png")

    ####News:
    wlttlmn = wordlistperlabel_generator(df, 'tweet_text_lemmatized', 1)
    wfttlmn = wordfrequency(wlttlmn)
    wcttlmn = wordcloud_generator(wfttlmn)
    wcttlmn.to_file("wordcloud/data-twitter/training/wordcloud_tweet_text_lemmatized_news-01.png")

    ####Opinion:
    wlttlmo = wordlistperlabel_generator(df, 'tweet_text_lemmatized', 0)
    wfttlmo = wordfrequency(wlttlmo)
    wcttlmo = wordcloud_generator(wfttlmo)
    wcttlmo.to_file("wordcloud/data-twitter/training/wordcloud_tweet_text_lemmatized_opinion-01.png")
    return

##Plotagem de gráficos da média da quantidade de caractéres por classe
def plot_mean_charcount(df):
    #####Gráficos de média de caracteres:
    ######Tweet_text_charcount:
    #ttcc_freq = calculate_mean(df, 'tweet_text_charcount')
    #ttcc_plt = bar_simple(ttcc_freq, 'Classes', 'Média de Caractere', 'Média de Caractere por Classes - Tuíte Original')
    #ttcc_plt.show()
    #ttcc_plt.savefig('matplotlib/data-twitter/training/mean_charcount/tweet_text_mean_charcount-01.png', format='png')
    #ttcc_plt.close()

    ######Tweet_text_lower_charcount
    #ttlwcc_freq = calculate_mean(df, 'tweet_text_lower_charcount')
    #ttlwcc_plt = bar_simple(ttlwcc_freq, 'Classes', 'Média de Caractere', 'Média de Caractere por Classes - Tuíte com Letras Minúsculas')
    #ttlwcc_plt.show()
    #ttlwcc_plt.savefig('matplotlib/data-twitter/training/mean_charcount/tweet_text_lower_mean_charcount-01.png', format='png')
    #ttlwcc_plt.close()

    ######Tweet_text_stemmed_charcount
    #ttstcc_freq = calculate_mean(df, 'tweet_text_stemmed_charcount')
    #ttstcc_plt = bar_simple(ttstcc_freq, 'Classes', 'Média de Caractere', 'Média de Caractere por Classes - Tuíte com Stemming')
    #ttstcc_plt.show()
    #ttstcc_plt.savefig('matplotlib/data-twitter/training/mean_charcount/tweet_text_stemmed_mean_charcount-01.png', format='png')
    #ttstcc_plt.close()

    ######Tweet_text_lemmatized_charcount
    #ttlmcc_freq = calculate_mean(df, 'tweet_text_lemmatized_charcount')
    #ttlmcc_plt = bar_simple(ttlmcc_freq, 'Classes', 'Média de Caractere', 'Média de Caractere por Classes - Tuíte com Lemmatization')
    #ttlmcc_plt.show()
    #ttlmcc_plt.savefig('matplotlib/data-twitter/training/mean_charcount/tweet_text_lemmatized_mean_charcount-01.png', format='png')
    #ttlmcc_plt.close()

    ######Tweet_text_spellchecked_charcount
    ttsccc_freq = calculate_mean(df, 'tweet_text_spellchecked_charcount')
    ttsccc_plt = bar_simple(ttsccc_freq, 'Classes', 'Média de Caractere', 'Média de Caractere por Classes - Tuíte com Spell Check')
    #ttsccc_plt.show()
    ttsccc_plt.savefig('matplotlib/data-twitter/training/mean_charcount/tweet_text_spellchecked_mean_charcount-01.png', format='png')
    ttsccc_plt.close()

    ######Tweet_text_spellchecked_lower_charcount
    ttsclcc_freq = calculate_mean(df, 'tweet_text_spellchecked_lower_charcount')
    ttsclcc_plt = bar_simple(ttsclcc_freq, 'Classes', 'Média de Caractere', 'Média de Caractere por Classes - Tuíte com Spell Check e Letras Minúsculas')
    #ttsclcc_plt.show()
    ttsclcc_plt.savefig('matplotlib/data-twitter/training/mean_charcount/tweet_text_spellchecked_lower_mean_charcount-01.png', format='png')
    ttsclcc_plt.close()

    ######Tweet_text_spellchecked_stemmed_charcount
    ttscsmcc_freq = calculate_mean(df, 'tweet_text_spellchecked_stemmed_charcount')
    ttscsmcc_plt = bar_simple(ttscsmcc_freq, 'Classes', 'Média de Caractere', 'Média de Caractere por Classes - Tuíte com Spell Check e Stemming')
    #ttscsmcc_plt.show()
    ttscsmcc_plt.savefig('matplotlib/data-twitter/training/mean_charcount/tweet_text_spellchecked_stemmed_mean_charcount-01.png', format='png')
    ttscsmcc_plt.close()

    ######Tweet_text_spellchecked_lemmatized_charcount
    ttsclmcc_freq = calculate_mean(df, 'tweet_text_spellchecked_lemmatized_charcount')
    ttsclmcc_plt = bar_simple(ttsclmcc_freq, 'Classes', 'Média de Caractere', 'Média de Caractere por Classes - Tuíte com Spell Check e Lemmatization')
    #ttsclmcc_plt.show()
    ttsclmcc_plt.savefig('matplotlib/data-twitter/training/mean_charcount/tweet_text_spellchecked_lemmatized_mean_charcount-01.png', format='png')
    ttsclmcc_plt.close()

    return

##Plotagem de gráficos da média da quantidade de palavras por classes
def plot_mean_wordcount(df):
    #####Gráficos de média de palavras:
    ######Tweet_text_wordcount:
    #ttwc_freq = calculate_mean(df, 'tweet_text_wordcount')
    #ttwc_plt = bar_simple(ttwc_freq, 'Classes', 'Média de Palavras', 'Média de Palavras por Classes - Tuíte Original')
    #ttwc_plt.show()
    #ttwc_plt.savefig('matplotlib/data-twitter/training/mean_wordcount/tweet_text_mean_wordcount-01.png', format='png')
    #ttwc_plt.close()

    ######Tweet_text_lower_wordcount
    #ttlwwc_freq = calculate_mean(df, 'tweet_text_lower_wordcount')
    #ttlwwc_plt = bar_simple(ttlwwc_freq, 'Classes', 'Média de Palavras', 'Média de Palavras por Classes - Tuíte com Letras Minúsculas')
    #ttlwwc_plt.show()
    #ttlwwc_plt.savefig('matplotlib/data-twitter/training/mean_wordcount/tweet_text_lower_mean_wordcount-01.png', format='png')
    #ttlwwc_plt.close()

    ######Tweet_text_stemmed_wordcount
    #ttstwc_freq = calculate_mean(df, 'tweet_text_stemmed_wordcount')
    #ttstwc_plt = bar_simple(ttstwc_freq, 'Classes', 'Média de Palavras', 'Média de Palavras por Classes - Tuíte com Stemming')
    ##ttstwc_plt.show()
    #ttstwc_plt.savefig('matplotlib/data-twitter/training/mean_wordcount/tweet_text_stemmed_mean_wordcount-01.png', format='png')
    #ttstwc_plt.close()

    ######Tweet_text_lemmatized_wordcount
    #ttlmwc_freq = calculate_mean(df, 'tweet_text_lemmatized_wordcount')
    #ttlmwc_plt = bar_simple(ttlmwc_freq, 'Classes', 'Média de Palavras', 'Média de Palavras por Classes - Tuíte com Lemmatization')
    #ttlmwc_plt.show()
    #ttlmwc_plt.savefig('matplotlib/data-twitter/training/mean_wordcount/tweet_text_lemmatized_mean_wordcount-01.png', format='png')
    #ttlmwc_plt.close()

    ######Tweet_text_spellchecked_wordcount
    ttscwc_freq = calculate_mean(df, 'tweet_text_spellchecked_wordcount')
    ttscwc_plt = bar_simple(ttscwc_freq, 'Classes', 'Média de Palavras', 'Média de Palavras por Classes - Tuíte com Spell Checker')
    #ttscwc_plt.show()
    ttscwc_plt.savefig('matplotlib/data-twitter/training/mean_wordcount/tweet_text_spellchecked_mean_wordcount-01.png', format='png')
    ttscwc_plt.close()

    ######Tweet_text_spellchecked_lower_wordcount
    ttsclwc_freq = calculate_mean(df, 'tweet_text_spellchecked_lower_wordcount')
    ttsclwc_plt = bar_simple(ttsclwc_freq, 'Classes', 'Média de Palavras', 'Média de Palavras por Classes - Tuíte com Spell Checker e Letras Minúsculas')
    #ttsclwc_plt.show()
    ttsclwc_plt.savefig('matplotlib/data-twitter/training/mean_wordcount/tweet_text_spellchecked_lower_mean_wordcount-01.png', format='png')
    ttsclwc_plt.close()

    ######Tweet_text_spellchecked_stemmed_wordcount
    ttscsmwc_freq = calculate_mean(df, 'tweet_text_spellchecked_stemmed_wordcount')
    ttscsmwc_plt = bar_simple(ttscsmwc_freq, 'Classes', 'Média de Palavras', 'Média de Palavras por Classes - Tuíte com Spell Checker e Stemming')
    #ttscsmwc_plt.show()
    ttscsmwc_plt.savefig('matplotlib/data-twitter/training/mean_wordcount/tweet_text_spellchecked_stemmed_mean_wordcount-01.png', format='png')
    ttscsmwc_plt.close()

    ######Tweet_text_spellchecked_lemmatized_wordcount
    ttsclmwc_freq = calculate_mean(df, 'tweet_text_spellchecked_lemmatized_wordcount')
    ttsclmwc_plt = bar_simple(ttsclmwc_freq, 'Classes', 'Média de Palavras', 'Média de Palavras por Classes - Tuíte com Spell Checker e Lemmatization')
    #ttsclmwc_plt.show()
    ttsclmwc_plt.savefig('matplotlib/data-twitter/training/mean_wordcount/tweet_text_spellchecked_lemmatized_mean_wordcount-01.png', format='png')
    ttsclmwc_plt.close()

    return

##Plotagem de gráficos do desvio padrão da quantidade de caractéres por classe
def plot_std_charcount(df):
    #####Gráficos de média de caracteres:
    ######Tweet_text_charcount:
    #ttcc_freq = calculate_std(df, 'tweet_text_charcount')
    #ttcc_plt = bar_simple(ttcc_freq, 'Classes', 'Desvio Padrão da Quantidade de Caracteres', 'Desvio Padrão da Quantidade de Caracteres por Classes - Tuíte Original')
    #ttcc_plt.show()
    #ttcc_plt.savefig('matplotlib/data-twitter/training/std_charcount/tweet_text_std_charcount-01.png', format='png')
    #ttcc_plt.close()

    ######Tweet_text_lower_charcount
    #ttlwcc_freq = calculate_std(df, 'tweet_text_lower_charcount')
    #ttlwcc_plt = bar_simple(ttlwcc_freq, 'Classes', 'Desvio Padrão da Quantidade de Caracteres', 'Desvio Padrão da Quantidade de Caracteres por Classes - Tuíte com Letras Minúsculas')
    #ttlwcc_plt.show()
    #ttlwcc_plt.savefig('matplotlib/data-twitter/training/std_charcount/tweet_text_lower_std_charcount-01.png', format='png')
    #ttlwcc_plt.close()

    ######Tweet_text_stemmed_charcount
    #ttstcc_freq = calculate_std(df, 'tweet_text_stemmed_charcount')
    #ttstcc_plt = bar_simple(ttstcc_freq, 'Classes', 'Desvio Padrão da Quantidade de Caracteres', 'Desvio Padrão da Quantidade de Caracteres por Classes - Tuíte com Stemming')
    #ttstcc_plt.show()
    #ttstcc_plt.savefig('matplotlib/data-twitter/training/std_charcount/tweet_text_stemmed_std_charcount-01.png', format='png')
    #ttstcc_plt.close()

    ######Tweet_text_lemmatized_charcount
    #ttlmcc_freq = calculate_std(df, 'tweet_text_lemmatized_charcount')
    #ttlmcc_plt = bar_simple(ttlmcc_freq, 'Classes', 'Desvio Padrão da Quantidade de Caracteres', 'Desvio Padrão da Quantidade de Caracteres por Classes - Tuíte com Lemmatization')
    #ttlmcc_plt.show()
    #ttlmcc_plt.savefig('matplotlib/data-twitter/training/std_charcount/tweet_text_lemmatized_std_charcount-01.png', format='png')
    #ttlmcc_plt.close()

    ######Tweet_text_spellchecked_charcount
    ttsccc_freq = calculate_std(df, 'tweet_text_spellchecked_charcount')
    ttsccc_plt = bar_simple(ttsccc_freq, 'Classes', 'Desvio Padrão da Quantidade de Caracteres', 'Desvio Padrão da Quantidade de Caracteres por Classes - Tuíte com Spell Checker')
    #ttsccc_plt.show()
    ttsccc_plt.savefig('matplotlib/data-twitter/training/std_charcount/tweet_text_spellchecked_std_charcount-01.png', format='png')
    ttsccc_plt.close()

    ######Tweet_text_spellchecked_lower_charcount
    ttsclcc_freq = calculate_std(df, 'tweet_text_spellchecked_lower_charcount')
    ttsclcc_plt = bar_simple(ttsclcc_freq, 'Classes', 'Desvio Padrão da Quantidade de Caracteres', 'Desvio Padrão da Quantidade de Caracteres por Classes - Tuíte com Spell Checker e Letras Minúsculas')
    #ttsclcc_plt.show()
    ttsclcc_plt.savefig('matplotlib/data-twitter/training/std_charcount/tweet_text_spellchecked_lower_std_charcount-01.png', format='png')
    ttsclcc_plt.close()

    ######Tweet_text_spellchecked_stemmed_charcount
    ttscsmcc_freq = calculate_std(df, 'tweet_text_spellchecked_stemmed_charcount')
    ttscsmcc_plt = bar_simple(ttscsmcc_freq, 'Classes', 'Desvio Padrão da Quantidade de Caracteres', 'Desvio Padrão da Quantidade de Caracteres por Classes - Tuíte com Spell Checker e Stemming')
    #tscsmcc_plt.show()
    ttscsmcc_plt.savefig('matplotlib/data-twitter/training/std_charcount/tweet_text_spellchecked_stemmed_std_charcount-01.png', format='png')
    ttscsmcc_plt.close()

    ######Tweet_text_spellchecked_lemmatized_charcount
    ttsclmcc_freq = calculate_std(df, 'tweet_text_spellchecked_lemmatized_charcount')
    ttsclmcc_plt = bar_simple(ttsclmcc_freq, 'Classes', 'Desvio Padrão da Quantidade de Caracteres', 'Desvio Padrão da Quantidade de Caracteres por Classes - Tuíte com Spell Checker e Lemmatization')
    #ttsclmcc_plt.show()
    ttsclmcc_plt.savefig('matplotlib/data-twitter/training/std_charcount/tweet_text_spellchecked_lemmatized_std_charcount-01.png', format='png')
    ttsclmcc_plt.close()

    return

##Plotagem de gráficos do desvio padrão da quantidade de palavras por classes
def plot_std_wordcount(df):
    #####Gráficos de média de palavras:
    ######Tweet_text_wordcount:
    #ttwc_freq = calculate_std(df, 'tweet_text_wordcount')
    #ttwc_plt = bar_simple(ttwc_freq, 'Classes', 'Desvio Padrão da Quantidade de Palavras', 'Desvio Padrão da Quantidade de Palavras por Classes - Tuíte Original')
    #ttwc_plt.show()
    #ttwc_plt.savefig('matplotlib/data-twitter/training/std_wordcount/tweet_text_std_wordcount-01.png', format='png')
    #ttwc_plt.close()

    ######Tweet_text_lower_wordcount
    #ttlwwc_freq = calculate_std(df, 'tweet_text_lower_wordcount')
    #ttlwwc_plt = bar_simple(ttlwwc_freq, 'Classes', 'Desvio Padrão da Quantidade de Palavras', 'Desvio Padrão da Quantidade de Palavras por Classes - Tuíte com Letras Minúsculas')
    #ttlwwc_plt.show()
    #ttlwwc_plt.savefig('matplotlib/data-twitter/training/std_wordcount/tweet_text_lower_std_wordcount-01.png', format='png')
    #ttlwwc_plt.close()

    ######Tweet_text_stemmed_wordcount
    #ttstwc_freq = calculate_std(df, 'tweet_text_stemmed_wordcount')
    #ttstwc_plt = bar_simple(ttstwc_freq, 'Classes', 'Desvio Padrão da Quantidade de Palavras', 'Desvio Padrão da Quantidade de Palavras por Classes - Tuíte com Stemming')
    #ttstwc_plt.show()
    #ttstwc_plt.savefig('matplotlib/data-twitter/training/std_wordcount/tweet_text_stemmed_std_wordcount-01.png', format='png')
    #ttstwc_plt.close()

    ######Tweet_text_lemmatized_wordcount
    #ttlmwc_freq = calculate_std(df, 'tweet_text_lemmatized_wordcount')
    #ttlmwc_plt = bar_simple(ttlmwc_freq, 'Classes', 'Desvio Padrão da Quantidade de Palavras', 'Desvio Padrão da Quantidade de Palavras por Classes - Tuíte com Lemmatization')
    #ttlmwc_plt.show()
    #ttlmwc_plt.savefig('matplotlib/data-twitter/training/std_wordcount/tweet_text_lemmatized_std_wordcount-01.png', format='png')
    #ttlmwc_plt.close()

    ######Tweet_text_spellchecked_wordcount
    ttscwc_freq = calculate_std(df, 'tweet_text_spellchecked_wordcount')
    ttscwc_plt = bar_simple(ttscwc_freq, 'Classes', 'Desvio Padrão da Quantidade de Palavras', 'Desvio Padrão da Quantidade de Palavras por Classes - Tuíte com Spell Checker')
    #ttscwc_plt.show()
    ttscwc_plt.savefig('matplotlib/data-twitter/training/std_wordcount/tweet_text_spellchecked_std_wordcount-01.png', format='png')
    ttscwc_plt.close()

    ######Tweet_text_spellchecked_lower_wordcount
    ttsclwc_freq = calculate_std(df, 'tweet_text_spellchecked_lower_wordcount')
    ttsclwc_plt = bar_simple(ttsclwc_freq, 'Classes', 'Desvio Padrão da Quantidade de Palavras', 'Desvio Padrão da Quantidade de Palavras por Classes - Tuíte com Spell Checker e Letras Minúsculas')
    #ttsclwc_plt.show()
    ttsclwc_plt.savefig('matplotlib/data-twitter/training/std_wordcount/tweet_text_spellchecked_lower_std_wordcount-01.png', format='png')
    ttsclwc_plt.close()

    ######Tweet_text_spellchecked_stemmed_wordcount
    ttscsmwc_freq = calculate_std(df, 'tweet_text_spellchecked_stemmed_wordcount')
    ttscsmwc_plt = bar_simple(ttscsmwc_freq, 'Classes', 'Desvio Padrão da Quantidade de Palavras', 'Desvio Padrão da Quantidade de Palavras por Classes - Tuíte com Spell Checker e Stemming')
    #ttscsmwc_plt.show()
    ttscsmwc_plt.savefig('matplotlib/data-twitter/training/std_wordcount/tweet_text_spellchecked_stemmed_std_wordcount-01.png', format='png')
    ttscsmwc_plt.close()

    ######Tweet_text_spellchecked_lemmatized_wordcount
    ttsclmwc_freq = calculate_std(df, 'tweet_text_spellchecked_lemmatized_wordcount')
    ttsclmwc_plt = bar_simple(ttsclmwc_freq, 'Classes', 'Desvio Padrão da Quantidade de Palavras', 'Desvio Padrão da Quantidade de Palavras por Classes - Tuíte com Spell Checker e Lemmatization')
    #ttsclmwc_plt.show()
    ttsclmwc_plt.savefig('matplotlib/data-twitter/training/std_wordcount/tweet_text_spellchecked_lemmatized_std_wordcount-01.png', format='png')
    ttsclmwc_plt.close()

    return

##Plotagem de gráficos sobre uma base de dados
def plot_database_exploratory_analysis(df):
    #####Gráfico de turnos:
    ts_list = timeshift_list_generator(df)
    ts_freq = frequency_generator(ts_list)
    ts_plt = bar_simple(ts_freq, 'Turno', 'Quantidade de Postagens', 'Quantidade de Postagens por Turno')
    #ts_plt.show()
    ts_plt.savefig('matplotlib/data-twitter/training/timeshift_frequency-01.png', format='png')
    #ts_plt.savefig('matplotlib/data-twitter/timeshift_frequency-02.png', format='png')
    ts_plt.close()

    #####Gráfico de nomes de usuários:
    usn_list = user_screen_name_list_generator(df)
    #usn_freq = frequency_generator(usn_list)
    usn_freq = frequency_generator_top10(usn_list)
    #usn_plt = bar_simple(usn_freq, 'Usuário', 'Quantidade de Postagens', 'Quantidade de Postagens por Usuário')
    usn_plt = bar_simple(usn_freq, 'Usuário', 'Quantidade de Postagens', 'TOP 10: Quantidade de Postagens por Usuário')
    #usn_plt.show()
    usn_plt.savefig('matplotlib/data-twitter/training/user_frequency-01.png', format='png')
    #usn_plt.savefig('matplotlib/data-twitter/user_frequency-02.png', format='png')
    usn_plt.close()

    #####Gráfico de estados:
    stt_list = state_list_generator(df)
    stt_freq = frequency_generator(stt_list)
    stt_plt = barh_simple(stt_freq, 'Estado', 'Quantidade de Postagens', 'Quantidade de Postagens por Estado')
    #stt_plt.show()
    stt_plt.savefig('matplotlib/data-twitter/training/state_frequency-01.png', format='png')
    #stt_plt.savefig('matplotlib/data-twitter/state_frequency-02.png', format='png')
    stt_plt.close()

    #####Gráfico de regiões:
    rgn_list = region_list_generator(df)
    rgn_freq = frequency_generator(rgn_list)
    rgn_plt = bar_simple(rgn_freq, 'Região', 'Quantidade de Postagens', 'Quantidade de Postagens por Região')
    #rgn_plt.show()
    rgn_plt.savefig('matplotlib/data-twitter/training/region_frequency-01.png', format='png')
    #rgn_plt.savefig('matplotlib/data-twitter/region_frequency-02.png', format='png')
    rgn_plt.close()
    return

##Ploatagem de gráficos considerando o rótulo dos tuítes em uma base de dados.
def plot_databaseperlabel_exploratory_analysis(df):
    #####Gráfico de rótulo por turno:
    #lpts_list = labelpertimeshift(df)
    #lpts_plt = bar_triple(lpts_list, 'Turnos', 'Quantidade de Postagens', 'Quantidade de Postagens por Turnos')
    #lpts_plt.show()
    #lpts_plt.savefig('matplotlib/data-twitter/training/labelpertimeshift-02.png', format='png')
    #lpts_plt.close()

    #####Gráfico de rótulo por região:
    #lprgn_list = labelperregion(df)
    #lprgn_plt = bar_triplep(lprgn_list, 'Região', 'Quantidade de Postagens', 'Quantidade de Postagens por Região')
    #lprgn_plt.show()
    #lprgn_plt.savefig('matplotlib/data-twitter/training/labelperregion-02.png', format='png')
    #lprgn_plt.close()

    #####Gráfico de rótulo por nome de usuário:
    #lpusn_list = labelperuserscreenname(df)
    #lpusn_plt = bar_triple(lpusn_list, 'Usuário', 'Quantidade de Postagens', 'TOP 10: Quantidade de Postagens por Usuário')
    #lpusn_plt.show()
    #lpusn_plt.savefig('matplotlib/data-twitter/training/labelperuserscreenname-01.png', format='png')
    #lpusn_plt.close()

    #####Gráfico de distribuição de classes:
    #cdstb_list = class_distribution(dfA, dfB)
    #cdstb_plt = bar_double(cdstb_list, 'Classes', 'Quantidade de Postagens', 'Quantidade de Postagens por Classes')
    #cdstb_plt.show()
    #cdstb_plt.savefig('matplotlib/data-twitter/training/class_distribution-01.png', format='png')
    #cdstb_plt.close()
    return

##Criar de colunas NAVA:
def nava_column_creator(df):
    ##Tweet_text:
    ttposlist = morphologic_analysis(df, 'tweet_text')
    ttposdict = navadictmaker(ttposlist)
    try:
        df.insert(25,'tweet_text_nava_substativos', ttposdict['Substantivos'])
    except:
        df['tweet_text_nava_substantivos'] = ttposdict['Substantivos']
    try:
        df.insert(26,'tweet_text_nava_adjetivos', ttposdict['Adjetivos'])
    except:
        df['tweet_text_nava_adjetivos'] = ttposdict['Adjetivos']
    try:
        df.insert(27,'tweet_text_nava_verbos', ttposdict['Verbos'])
    except:
        df['tweet_text_nava_verbos'] = ttposdict['Verbos']
    try:
        df.insert(28,'tweet_text_nava_adverbios', ttposdict['Adverbios'])
    except:
        df['tweet_text_nava_adverbios'] = ttposdict['Adverbios']

    ##Tweet_text_lower:
    ttposlist = morphologic_analysis(df, 'tweet_text_lower')
    ttposdict = navadictmaker(ttposlist)
    try:
        df.insert(29,'tweet_text_lower_nava_substativos', ttposdict['Substantivos'])
    except:
        df['tweet_text_lower_nava_substantivos'] = ttposdict['Substantivos']
    try:
        df.insert(30,'tweet_text_lower_nava_adjetivos', ttposdict['Adjetivos'])
    except:
        df['tweet_text_lower_nava_adjetivos'] = ttposdict['Adjetivos']
    try:
        df.insert(31,'tweet_text_lower_nava_verbos', ttposdict['Verbos'])
    except:
        df['tweet_text_lower_nava_verbos'] = ttposdict['Verbos']
    try:
        df.insert(32,'tweet_text_lower_nava_adverbios', ttposdict['Adverbios'])
    except:
        df['tweet_text_lower_nava_adverbios'] = ttposdict['Adverbios']

    ##Tweet_text_stemmed:
    ttposlist = morphologic_analysis(df, 'tweet_text_stemmed')
    ttposdict = navadictmaker(ttposlist)
    try:
        df.insert(33,'tweet_text_stemmed_nava_substativos', ttposdict['Substantivos'])
    except:
        df['tweet_text_stemmed_nava_substantivos'] = ttposdict['Substantivos']
    try:
        df.insert(34,'tweet_text_stemmed_nava_adjetivos', ttposdict['Adjetivos'])
    except:
        df['tweet_text_stemmed_nava_adjetivos'] = ttposdict['Adjetivos']
    try:
        df.insert(35,'tweet_text_stemmed_nava_verbos', ttposdict['Verbos'])
    except:
        df['tweet_text_stemmed_nava_verbos'] = ttposdict['Verbos']
    try:
        df.insert(36,'tweet_text_stemmed_nava_adverbios', ttposdict['Adverbios'])
    except:
        df['tweet_text_stemmed_nava_adverbios'] = ttposdict['Adverbios']

    ##Tweet_text_lemmatized:
    ttposlist = morphologic_analysis(df, 'tweet_text_lemmatized')
    ttposdict = navadictmaker(ttposlist)
    try:
        df.insert(37,'tweet_text_lemmatized_nava_substativos', ttposdict['Substantivos'])
    except:
        df['tweet_text_lemmatized_nava_substantivos'] = ttposdict['Substantivos']
    try:
        df.insert(38,'tweet_text_lemmatized_nava_adjetivos', ttposdict['Adjetivos'])
    except:
        df['tweet_text_lemmatized_nava_adjetivos'] = ttposdict['Adjetivos']
    try:
        df.insert(39,'tweet_text_lemmatized_nava_verbos', ttposdict['Verbos'])
    except:
        df['tweet_text_lemmatized_nava_verbos'] = ttposdict['Verbos']
    try:
        df.insert(40,'tweet_text_lemmatized_nava_adverbios', ttposdict['Adverbios'])
    except:
        df['tweet_text_lemmatized_nava_adverbios'] = ttposdict['Adverbios']

    ##Tweet_text_spellchecked:
    ttposlist = morphologic_analysis(df, 'tweet_text_spellchecked')
    ttposdict = navadictmaker(ttposlist)
    try:
        df.insert(41,'tweet_text_spellchecked_nava_substativos', ttposdict['Substantivos'])
    except:
        df['tweet_text_spellchecked_nava_substantivos'] = ttposdict['Substantivos']
    try:
        df.insert(42,'tweet_text_spellchecked_nava_adjetivos', ttposdict['Adjetivos'])
    except:
        df['tweet_text_spellchecked_nava_adjetivos'] = ttposdict['Adjetivos']
    try:
        df.insert(43,'tweet_text_spellchecked_nava_verbos', ttposdict['Verbos'])
    except:
        df['tweet_text_spellchecked_nava_verbos'] = ttposdict['Verbos']
    try:
        df.insert(44,'tweet_text_spellchecked_nava_adverbios', ttposdict['Adverbios'])
    except:
        df['tweet_text_spellchecked_nava_adverbios'] = ttposdict['Adverbios']
    
    ##Tweet_text_spellchecked_lower:
    ttposlist = morphologic_analysis(df, 'tweet_text_spellchecked_lower')
    ttposdict = navadictmaker(ttposlist)
    try:
        df.insert(45,'tweet_text_spellchecked_lower_nava_substativos', ttposdict['Substantivos'])
    except:
        df['tweet_text_spellchecked_lower_nava_substantivos'] = ttposdict['Substantivos']
    try:
        df.insert(46,'tweet_text_spellchecked_lower_nava_adjetivos', ttposdict['Adjetivos'])
    except:
        df['tweet_text_spellchecked_lower_nava_adjetivos'] = ttposdict['Adjetivos']
    try:
        df.insert(47,'tweet_text_spellchecked_lower_nava_verbos', ttposdict['Verbos'])
    except:
        df['tweet_text_spellchecked_lower_nava_verbos'] = ttposdict['Verbos']
    try:
        df.insert(48,'tweet_text_spellchecked_lower_nava_adverbios', ttposdict['Adverbios'])
    except:
        df['tweet_text_spellchecked_lower_nava_adverbios'] = ttposdict['Adverbios']

    ##Tweet_text_spellchecked_stemmed:
    ttposlist = morphologic_analysis(df, 'tweet_text_spellchecked_stemmed')
    ttposdict = navadictmaker(ttposlist)
    try:
        df.insert(49,'tweet_text_spellchecked_stemmed_nava_substativos', ttposdict['Substantivos'])
    except:
        df['tweet_text_spellchecked_stemmed_nava_substantivos'] = ttposdict['Substantivos']
    try:
        df.insert(50,'tweet_text_spellchecked_stemmed_nava_adjetivos', ttposdict['Adjetivos'])
    except:
        df['tweet_text_spellchecked_stemmed_nava_adjetivos'] = ttposdict['Adjetivos']
    try:
        df.insert(51,'tweet_text_spellchecked_stemmed_nava_verbos', ttposdict['Verbos'])
    except:
        df['tweet_text_spellchecked_stemmed_nava_verbos'] = ttposdict['Verbos']
    try:
        df.insert(52,'tweet_text_spellchecked_stemmed_nava_adverbios', ttposdict['Adverbios'])
    except:
        df['tweet_text_spellchecked_stemmed_nava_adverbios'] = ttposdict['Adverbios']

    ##Tweet_text_spellchecked_lemmatized:
    ttposlist = morphologic_analysis(df, 'tweet_text_spellchecked_lemmatized')
    ttposdict = navadictmaker(ttposlist)
    try:
        df.insert(53,'tweet_text_spellchecked_lemmatized_nava_substativos', ttposdict['Substantivos'])
    except:
        df['tweet_text_spellchecked_lemmatized_nava_substantivos'] = ttposdict['Substantivos']
    try:
        df.insert(54,'tweet_text_spellchecked_lemmatized_nava_adjetivos', ttposdict['Adjetivos'])
    except:
        df['tweet_text_spellchecked_lemmatized_nava_adjetivos'] = ttposdict['Adjetivos']
    try:
        df.insert(55,'tweet_text_spellchecked_lemmatized_nava_verbos', ttposdict['Verbos'])
    except:
        df['tweet_text_spellchecked_lemmatized_nava_verbos'] = ttposdict['Verbos']
    try:
        df.insert(56,'tweet_text_spellchecked_lemmatized_nava_adverbios', ttposdict['Adverbios'])
    except:
        df['tweet_text_spellchecked_lemmatized_nava_adverbios'] = ttposdict['Adverbios']

    return

##Criação de novas colunas
def column_creator(df):
    ####Criação de colunas:
    timeshift_listing(df)
    user_crawler(df)
    location_trater(df)
    geopy_stateregion(df)
    
    return

##Criação de colunas de quantidade de caracteres
def charcounter(df):
    #tcc = tweet_charcounter(df, 'tweet_text')
    #try:
        #df.insert(9,'tweet_text_charcount', tcc)
    #except:
        #df['tweet_text_charcount'] = tcc
    
    #tlcc = tweet_charcounter(df, 'tweet_text_lower')
    #try:
        #df.insert(10,'tweet_text_lower_charcount', tlcc)
    #except:
        #df['tweet_text_lower_charcount'] = tlcc
    
    #tsmcc = tweet_charcounter(df, 'tweet_text_stemmed')
    #try:
        #df.insert(11,'tweet_text_stemmed_charcount', tmscc)
    #except:
        #df['tweet_text_stemmed_charcount'] = tmscc
    
    #tlmcc = tweet_charcounter(df, 'tweet_text_lemmatized')
    #try:
        #df.insert(12,'tweet_text_lemmatized_charcount', tlmcc)
    #except:
        #df['tweet_text_lemmatized_charcount'] = tlmcc
    
    tsccc = tweet_charcounter(df, 'tweet_text_spellchecked')
    try:
        df.insert(13,'tweet_text_spellchecked_charcount', tsccc)
    except:
        df['tweet_text_spellchecked_charcount'] = tsccc
    
    tsclcc = tweet_charcounter(df, 'tweet_text_spellchecked_lower')
    try:
        df.insert(14,'tweet_text_spellchecked_lower_charcount', tsclcc)
    except:
        df['tweet_text_spellchecked_lower_charcount'] = tsclcc
    
    tscsmcc = tweet_charcounter(df, 'tweet_text_spellchecked_stemmed')
    try:
        df.insert(15,'tweet_text_spellchecked_stemmed_charcount', tscsmcc)
    except:
        df['tweet_text_spellchecked_stemmed_charcount'] = tscsmcc

    tsclmcc = tweet_charcounter(df, 'tweet_text_spellchecked_lemmatized')
    try:
        df.insert(16,'tweet_text_spellchecked_lemmatized_charcount', tsclmcc)
    except:
        df['tweet_text_spellchecked_lemmatized_charcount'] = tsclmcc
    
    return

##Criaçao de colunas de quantidade de palavras
def wordcounter(df):
    #twc = tweet_wordcounter(df, 'tweet_text')
    #try:
        #df.insert(13,'tweet_text_wordcount', twc)
    #except:
        #df['tweet_text_wordcount'] = twc
    
    #tlwc = tweet_wordcounter(df, 'tweet_text_lower')
    #try:
        #df.insert(14,'tweet_text_lower_wordcount', tlwc)
    #except:
        #df['tweet_text_lower_wordcount'] = tlwc
    
    #tsmwc = tweet_wordcounter(df, 'tweet_text_stemmed')
    #try:
        #df.insert(15,'tweet_text_stemmed_wordcount', tsmwc)
    #except:
        #df['tweet_text_stemmed_wordcount'] = tsmwc
    
    #tlmwc = tweet_wordcounter(df, 'tweet_text_lemmatized')
    #try:
        #df.insert(16,'tweet_text_lemmatized_wordcount', tlmwc)
    #except:
        #df['tweet_text_lemmatized_wordcount'] = tlmwc
    
    tscwc = tweet_wordcounter(df, 'tweet_text_spellchecked')
    try:
        df.insert(21,'tweet_text_spellchecked_wordcount', tscwc)
    except:
        df['tweet_text_spellchecked_wordcount'] = tscwc

    tsclwc = tweet_wordcounter(df, 'tweet_text_spellchecked_lower')
    try:
        df.insert(22,'tweet_text_spellchecked_lower_wordcount', tsclwc)
    except:
        df['tweet_text_spellchecked_lower_wordcount'] = tsclwc
    
    tscsmwc = tweet_wordcounter(df, 'tweet_text_spellchecked_stemmed')
    try:
        df.insert(23,'tweet_text_spellchecked_stemmed_wordcount', tscsmwc)
    except:
        df['tweet_text_spellchecked_stemmed_wordcount'] = tscsmwc
    
    tsclmwc = tweet_wordcounter(df, 'tweet_text_spellchecked_lemmatized')
    try:
        df.insert(24,'tweet_text_spellchecked_lemmatized_wordcount', tsclmwc)
    except:
        df['tweet_text_spellchecked_lemmatized_wordcount'] = tsclmwc

    return

##Função main:
###Abertura de arquivo e criação do dataframe:
df_file = pd.read_csv("data/data-twitter/training/rotulaçao[iguais]_complete.csv", sep=";")
df = pd.DataFrame(df_file)
#df = df.drop_duplicates(subset='tweet_id', keep='first')

#df_file = pd.read_csv("data/data-twitter/data-twitter_upgraded.csv", sep=";")
#df = pd.DataFrame(df_file)
#df = df.drop_duplicates(subset='tweet_id', keep='first')

#dfA_file = pd.read_csv("data/data-twitter/training/rotulaçao[anisiofilho].csv", sep=",")
#dfA = pd.DataFrame(dfA_file)
#dfA = dfA.drop_duplicates(subset='tweet_id', keep='first')

#dfB_file = pd.read_csv("data/data-twitter/training/rotulaçao[debora].csv", sep=",")
#dfB = pd.DataFrame(dfB_file)
#dfB = dfB.drop_duplicates(subset='tweet_id', keep='first')

###Chamadas de funções:
#column_creator(df)
#charcounter(df)
#wordcounter(df)
nava_column_creator(df)
#plot_database_exploratory_analysis(df)
#plot_databaseperlabel_exploratory_analysis(df)
#plot_mean_charcount(df)
#plot_mean_wordcount(df)
#plot_std_charcount(df)
#plot_std_wordcount(df)
#wordcloud_wordlistperlabel_creator(df)

###Salvar alterações no csv:
add_file = df.to_csv("data/data-twitter/training/rotulaçao[iguais]_complete_NOVO.csv", sep=";", index=False)

#####Chamadas de funções de gráficos:
#frequency_generator(word_list)
#bar_simple(freqdict, xlabel, ylabel, title)
#bar_double(labeldict, xlabel, ylabel, title)
#bar_triple(labeldict, xlabel, ylabel, title)
#barh_simple(freqdict, ylabel, xlabel, title)

end = timeit.default_timer()
print ('Duração: %f segundos' % (end - start))