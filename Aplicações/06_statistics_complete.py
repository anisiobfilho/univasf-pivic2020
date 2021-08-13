import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import shapiro, normaltest, anderson, levene, ttest_ind, wilcoxon

import time
import timeit

start = timeit.default_timer()

#FUNCTIONS:
##STATISTICS TESTS:
def shapiro_wilk_test(data):
    # normality test
    stat, p = shapiro(data)
    print('\tStatistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('\tSample looks Gaussian (fail to reject H0)')
    else:
        print('\tSample does not look Gaussian (reject H0)')
    return p

def dagostino_test(data):
    # normality test
    stat, p = normaltest(data)
    print('\tStatistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('\tSample looks Gaussian (fail to reject H0)')
    else:
        print('\tSample does not look Gaussian (reject H0)')

def anderson_darling_test(data):
    # normality test
    result = anderson(data)
    print('\tStatistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('\t%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('\t%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

def levene_test(data1, data2):
    # normality test
    stat, p = levene(data1, data2)
    print('\tStatistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('\tThe difference between the means is not statistically significant (do not reject H0)')
    else:
        print('\tThe difference between the means is statistically significant (reject H0)')
    return

def tstudent_test(data1, data2):
    print("\tT-Student Test:")
    # normality test
    stat, p = ttest_ind(data1, data2, equal_var=True)
    print('\tStatistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('\tThe difference between the means is not statistically significant (do not reject H0)')
    else:
        print('\tThe difference between the means is statistically significant (reject H0)')
    return

def wilcoxon_test(data1, data2):
    print("\tWilcoxon Test:")
    # normality test
    stat, p = wilcoxon(data1, data2)
    print('\tStatistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('\tThe difference between the means is not statistically significant (do not reject H0)')
    else:
        print('\tThe difference between the means is statistically significant (reject H0)')
    return


def perform_fulltest(data1, data2, alpha1, alpha2):
    if alpha1 > 0.05 and alpha2 > 0.05:
        tstudent_test(data1, data2)
    else:
        wilcoxon_test(data1, data2)

    return


##MAIN:
#tree = [0.6326530612244898,0.5816326530612245,0.6428571428571429,0.5918367346938775,0.5408163265306123,0.5876288659793815,0.6288659793814433,0.6185567010309279,0.5979381443298969,0.5463917525773195]
#multinomialnb = [0.6836734693877551,0.7040816326530612,0.6632653061224489,0.6530612244897959,0.5714285714285714,0.7319587628865979,0.7319587628865979,0.6391752577319587,0.6391752577319587,0.6391752577319587]
#svc = [0.7346938775510204,0.7448979591836735,0.7040816326530612,0.7244897959183674,0.6122448979591837,0.7319587628865979,0.711340206185567,0.6597938144329897,0.6185567010309279,0.7010309278350515]
#randomforest = [0.6938775510204082,0.7040816326530612,0.7040816326530612,0.6428571428571429,0.5510204081632653,0.7010309278350515,0.7010309278350515,0.6185567010309279,0.6185567010309279,0.6494845360824743]


##DATA:
decisiontree = [0.6530612244897959,0.7142857142857143,0.6428571428571429,0.6224489795918368,0.6020408163265306,0.7010309278350515,0.6597938144329897,0.6082474226804123,0.5360824742268041,0.5773195876288659]
decisiontree_ov = [0.6938775510204082,0.6836734693877551,0.6632653061224489,0.6530612244897959,0.5102040816326531,0.6288659793814433,0.6082474226804123,0.5876288659793815,0.5979381443298969,0.6391752577319587]
decisiontree_ov_un = [0.7142857142857143,0.6428571428571429,0.6632653061224489,0.6224489795918368,0.5918367346938775,0.6701030927835051,0.6597938144329897,0.711340206185567,0.5567010309278351,0.5257731958762887]

multinomialnb = [0.6836734693877551,0.7040816326530612,0.6632653061224489,0.6530612244897959,0.5714285714285714,0.7319587628865979,0.7319587628865979,0.6391752577319587,0.6391752577319587,0.6391752577319587]
multinomialnb_ov = [0.5918367346938775,0.5,0.5204081632653061,0.6326530612244898,0.5816326530612245,0.5773195876288659,0.6701030927835051,0.5773195876288659,0.6391752577319587,0.5773195876288659]
multinomialnb_ov_un = [0.5510204081632653,0.5510204081632653,0.5204081632653061,0.6326530612244898,0.5510204081632653,0.6082474226804123,0.6701030927835051,0.5876288659793815,0.6288659793814433,0.5670103092783505]

svc = [0.7346938775510204,0.7448979591836735,0.7040816326530612,0.7244897959183674,0.6122448979591837,0.7319587628865979,0.711340206185567,0.6597938144329897,0.6185567010309279,0.7010309278350515]
svc_ov = [0.7448979591836735,0.7448979591836735,0.7040816326530612,0.7244897959183674,0.6122448979591837,0.711340206185567,0.711340206185567,0.6597938144329897,0.6185567010309279,0.711340206185567]
svc_ov_un = [0.7448979591836735,0.7346938775510204,0.7040816326530612,0.7142857142857143,0.6122448979591837,0.7319587628865979,0.711340206185567,0.6597938144329897,0.6185567010309279,0.7010309278350515]

randomforest = [0.6836734693877551,0.6836734693877551,0.7040816326530612,0.673469387755102,0.5816326530612245,0.711340206185567,0.7422680412371134,0.6288659793814433,0.6185567010309279,0.6494845360824743]
randomforest_ov = [0.7448979591836735,0.6632653061224489,0.7040816326530612,0.6938775510204082,0.6224489795918368,0.711340206185567,0.7628865979381443,0.6804123711340206,0.6701030927835051,0.711340206185567]
randomforest_ov_un = [0.6836734693877551,0.7142857142857143,0.6836734693877551,0.7040816326530612,0.6530612244897959,0.7319587628865979,0.7319587628865979,0.6494845360824743,0.6494845360824743,0.7216494845360825]

adaboost = [0.6632653061224489,0.6836734693877551,0.7244897959183674,0.6530612244897959,0.5204081632653061,0.6391752577319587,0.6288659793814433,0.6288659793814433,0.5670103092783505,0.6185567010309279]
adaboost_ov = [0.7448979591836735,0.5918367346938775,0.673469387755102,0.673469387755102,0.5816326530612245,0.6494845360824743,0.711340206185567,0.6494845360824743,0.6288659793814433,0.6288659793814433]
adaboost_ov_un = [0.7346938775510204,0.7244897959183674,0.6224489795918368,0.6428571428571429,0.5918367346938775,0.6804123711340206,0.6804123711340206,0.6804123711340206,0.6185567010309279,0.6082474226804123]

xgboost = [0.7142857142857143,0.6632653061224489,0.673469387755102,0.6632653061224489,0.6632653061224489,0.6701030927835051,0.711340206185567,0.6597938144329897,0.6082474226804123,0.6494845360824743]
xgboost_ov = [0.7244897959183674,0.673469387755102,0.6836734693877551,0.6836734693877551,0.7040816326530612,0.6907216494845361,0.7319587628865979,0.6494845360824743,0.6185567010309279,0.6494845360824743]
xgboost_ov_un = [0.7448979591836735,0.6836734693877551,0.6938775510204082,0.6836734693877551,0.673469387755102,0.6494845360824743,0.7319587628865979,0.6597938144329897,0.6185567010309279,0.6288659793814433]

#print("\n")
#print("Statistics Tests:")
#print("\n")
#print("Acuracy:")
#print("Decision Tree: 0.68")
#print("Multinomial Naive Bayes: 0.70")
#print("SVC: 0.73")
#print("Random Forest: 0.74")

#print("\n")
#print("Execution Time (in seconds):")
#print("Decision Tree: 5.113687")
#print("Multinomial Naive Bayes: 4.365976")
#print("SVC: 14.381937")
#print("Random Forest: 78.041038")

#print("\n")
print("Standard Deviaton:")
print("\n")
print("Desicion Tree:", np.std(decisiontree))
print("DT + OV:", np.std(decisiontree_ov))
print("DT + OV + UN:", np.std(decisiontree_ov_un))
print("\n")
print("Multinomial Naive Bayes:", np.std(multinomialnb))
print("MNB + OV:", np.std(multinomialnb_ov))
print("MNB + OV + UN:", np.std(multinomialnb_ov_un))
print("\n")
print("SVC:", np.std(svc))
print("SVC + OV:", np.std(svc_ov))
print("SVC + OV + UN:", np.std(svc_ov_un))
print("\n")
print("Random Forest:", np.std(randomforest))
print("RF + OV:", np.std(randomforest_ov))
print("RF + OV + UN:", np.std(randomforest_ov_un))
print("\n")
print("Adaboost:", np.std(adaboost))
print("Adaboost + OV:", np.std(adaboost_ov))
print("Adaboost + OV + UN:", np.std(adaboost_ov_un))
print("\n")
print("XGBoost:", np.std(xgboost))
print("XGBoost + OV:", np.std(xgboost_ov))
print("XGBoost + OV + UN:", np.std(xgboost_ov_un))
print("\n")
#Normality Tests:
print("\n")
print("Normality Tests:")

print("Shapiro-Wilk Tests:")
print("Decision Tree:")
decisiontree_alpha = shapiro_wilk_test(decisiontree)
print("DT + Oversampling:")
decisiontree_ov_alpha = shapiro_wilk_test(decisiontree_ov)
print("DT + Oversampling + Undersampling:")
decisiontree_ov_un_alpha = shapiro_wilk_test(decisiontree_ov_un)

print("Multinomial Naive Bayes:")
multinomialnb_alpha = shapiro_wilk_test(multinomialnb)
print("MNB + Oversampling:")
multinomialnb_ov_alpha = shapiro_wilk_test(multinomialnb_ov)
print("MNB + Oversampling + Undersampling:")
multinomialnb_ov_un_alpha = shapiro_wilk_test(multinomialnb_ov_un)

print("SVC:")
svc_alpha = shapiro_wilk_test(svc)
print("SVC + Oversampling:")
svc_ov_alpha = shapiro_wilk_test(svc_ov)
print("SVC + Oversampling + Undersampling:")
svc_ov_un_alpha = shapiro_wilk_test(svc_ov_un)

print("Random Forest:")
randomforest_alpha = shapiro_wilk_test(randomforest)
print("RF + Oversampling:")
randomforest_ov_alpha = shapiro_wilk_test(randomforest_ov)
print("RF + Oversampling + Undersampling:")
randomforest_ov_un_alpha = shapiro_wilk_test(randomforest_ov_un)

print("Adaboost:")
adaboost_alpha = shapiro_wilk_test(adaboost)
print("Adaboost + Oversampling:")
adaboost_ov_alpha = shapiro_wilk_test(adaboost_ov)
print("Adaboost + Oversampling + Undersampling:")
adaboost_ov_un_alpha = shapiro_wilk_test(adaboost_ov_un)

print("XGBoost:")
xgboost_alpha = shapiro_wilk_test(xgboost)
print("XGBoost + Oversampling:")
xgboost_ov_alpha = shapiro_wilk_test(xgboost_ov)
print("XGBoost + Oversampling + Undersampling:")
xgboost_ov_un_alpha = shapiro_wilk_test(xgboost_ov_un)


print("\n")
print("Comparações de um algoritmo em relação aos demais algoritmos:")
print("Testes comparando com: 'Random Forest + Oversampling + Undersampling'")
print("Decision Tree:")
perform_fulltest(randomforest_ov_un, decisiontree, randomforest_ov_un_alpha, decisiontree_alpha)
print("Decision Tree + Oversampling:")
perform_fulltest(randomforest_ov_un, decisiontree_ov, randomforest_ov_un_alpha, decisiontree_ov_alpha)
print("Decision Tree + Oversampling + Undersampling:")
perform_fulltest(randomforest_ov_un, decisiontree_ov_un, randomforest_ov_un_alpha, decisiontree_ov_un_alpha)
print("Multinomial Naive Bayes:")
perform_fulltest(randomforest_ov_un, multinomialnb, randomforest_ov_un_alpha, multinomialnb_alpha)
print("MNB + Oversampling:")
perform_fulltest(randomforest_ov_un, multinomialnb_ov, randomforest_ov_un_alpha, multinomialnb_ov_alpha)
print("MNB + Oversampling + Undersampling:")
perform_fulltest(randomforest_ov_un, multinomialnb_ov_un, randomforest_ov_un_alpha, multinomialnb_ov_un_alpha)
print("SVC:")
perform_fulltest(randomforest_ov_un, svc, randomforest_ov_un_alpha, svc_alpha)
print("SVC + Oversampling:")
perform_fulltest(randomforest_ov_un, svc_ov, randomforest_ov_un_alpha, svc_ov_alpha)
print("SVC + Oversampling + Undersampling:")
perform_fulltest(randomforest_ov_un, svc_ov_un, randomforest_ov_un_alpha, svc_ov_un_alpha)
print("Random Forest:")
perform_fulltest(randomforest_ov_un, randomforest, randomforest_ov_un_alpha, randomforest_alpha)
print("Random Forest + Oversampling:")
perform_fulltest(randomforest_ov_un, randomforest_ov, randomforest_ov_un_alpha, randomforest_ov_alpha)
print("Adaboost:")
perform_fulltest(randomforest_ov_un, adaboost, randomforest_ov_un_alpha, adaboost_alpha)
print("Adaboost + Oversampling:")
perform_fulltest(randomforest_ov_un, adaboost_ov, randomforest_ov_un_alpha, adaboost_ov_alpha)
print("Adaboost + Oversampling + Undersampling:")
perform_fulltest(randomforest_ov_un, adaboost_ov_un, randomforest_ov_un_alpha, adaboost_ov_un_alpha)
print("XGBoost:")
perform_fulltest(randomforest_ov_un, xgboost, randomforest_ov_un_alpha, xgboost_alpha)
print("XGBoost + Oversampling:")
perform_fulltest(randomforest_ov_un, xgboost_ov, randomforest_ov_un_alpha, xgboost_ov_alpha)
print("XGBoost + Oversampling + Undersampling:")
perform_fulltest(randomforest_ov_un, xgboost_ov_un, randomforest_ov_un_alpha, xgboost_ov_un_alpha)

print("\n")
print("Comparações de um algoritmo em relação aos demais algoritmos:")
print("Testes comparando com: 'Multinomial Naive Bayes")
print("Decision Tree:")
perform_fulltest(multinomialnb, decisiontree, multinomialnb_alpha, decisiontree_alpha)
print("Decision Tree + Oversampling:")
perform_fulltest(multinomialnb, decisiontree_ov, multinomialnb_alpha, decisiontree_ov_alpha)
print("Decision Tree + Oversampling + Undersampling:")
perform_fulltest(multinomialnb, decisiontree_ov_un, multinomialnb_alpha, decisiontree_ov_un_alpha)
print("MNB + Oversampling:")
perform_fulltest(multinomialnb, multinomialnb_ov, multinomialnb_alpha, multinomialnb_ov_alpha)
print("MNB + Oversampling + Undersampling:")
perform_fulltest(multinomialnb, multinomialnb_ov_un, multinomialnb_alpha, multinomialnb_ov_un_alpha)
print("SVC:")
perform_fulltest(multinomialnb, svc, multinomialnb_alpha, svc_alpha)
print("SVC + Oversampling:")
perform_fulltest(multinomialnb, svc_ov, multinomialnb_alpha, svc_ov_alpha)
print("SVC + Oversampling + Undersampling:")
perform_fulltest(multinomialnb, svc_ov_un, multinomialnb_alpha, svc_ov_un_alpha)
print("Random Forest:")
perform_fulltest(multinomialnb, randomforest, multinomialnb_alpha, randomforest_alpha)
print("Random Forest + Oversampling:")
perform_fulltest(multinomialnb, randomforest_ov, multinomialnb_alpha, randomforest_ov_alpha)
print("Random Forest + Oversampling + Undersampling:")
perform_fulltest(multinomialnb, randomforest_ov_un, multinomialnb_alpha, randomforest_ov_un_alpha)
print("Adaboost:")
perform_fulltest(multinomialnb, adaboost, multinomialnb_alpha, adaboost_alpha)
print("Adaboost + Oversampling:")
perform_fulltest(multinomialnb, adaboost_ov, multinomialnb_alpha, adaboost_ov_alpha)
print("Adaboost + Oversampling + Undersampling:")
perform_fulltest(multinomialnb, adaboost_ov_un, multinomialnb_alpha, adaboost_ov_un_alpha)
print("XGBoost:")
perform_fulltest(multinomialnb, xgboost, multinomialnb_alpha, xgboost_alpha)
print("XGBoost + Oversampling:")
perform_fulltest(multinomialnb, xgboost_ov, multinomialnb_alpha, xgboost_ov_alpha)
print("XGBoost + Oversampling + Undersampling:")
perform_fulltest(multinomialnb, xgboost_ov_un, multinomialnb_alpha, xgboost_ov_un_alpha)


end = timeit.default_timer()
print ('Duração: %f segundos' % (end - start))