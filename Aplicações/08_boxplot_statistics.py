import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import time
import timeit

start = timeit.default_timer()


def plot_boxplot(data, label_description):
    # Multiple box plots on one Axes
    fig, ax = plt.subplots(figsize=(16.00,9.00))
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set(
        axisbelow=True,  # Hide the grid behind plot objects
        #title='Comparação entre Algoritmos',
        xlabel='Algorithm',
        ylabel='Acuracy',
        )
    # Set the axes ranges and axes labels
    #ax.set_xlim(0.5, len(data) + 0.5)
    #top = 40
    #bottom = -5
    #ax.set_ylim(bottom, top)
    #ax.set_xticklabels(np.repeat(label_description, 2), rotation=0, fontsize=8)
    ax.set_xticklabels(label_description, fontsize=8)
    ax.boxplot(data)
    return plt


#Dados:
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


label_description = ['RF+OV+UN', 
                     'DT', 'DT+OV', 'DT+OV+UN', 
                     'MNB', 'MNB+OV', 'MNB+OV+UN', 
                     'SVC', 'SVC+OV', 'SVC+OV+UN', 
                     'RF', 'RF+OV', 
                     'ADA', 'ADA+OV', 'ADA+OV+UN', 
                     'XGB', 'XGB+OV', 'XGB+OV+UN'
                    ]
data1 = [randomforest_ov_un, 
         decisiontree, decisiontree_ov, decisiontree_ov_un,
         multinomialnb, multinomialnb_ov, multinomialnb_ov_un,
         svc, svc_ov, svc_ov_un,
         randomforest, randomforest_ov,
         adaboost, adaboost_ov, adaboost_ov_un,
         xgboost, xgboost_ov, xgboost_ov_un
        ]
plt = plot_boxplot(data1, label_description)
#plt.show()
plt.savefig('matplotlib/data-twitter/training/boxplot/rf+ov+un_vs_all.png', format='png')
plt.close()

label_description = ['MNB', 
                     'DT', 'DT+OV', 'DT+OV+UN', 
                     'MNB+OV', 'MNB+OV+UN', 
                     'SVC', 'SVC+OV', 'SVC+OV+UN', 
                     'RF', 'RF+OV', 'RF+OV+UN',
                     'ADA', 'ADA+OV', 'ADA+OV+UN', 
                     'XGB', 'XGB+OV', 'XGB+OV+UN'
                    ]
data2 = [multinomialnb, 
         decisiontree, decisiontree_ov, decisiontree_ov_un,
         multinomialnb_ov, multinomialnb_ov_un,
         svc, svc_ov, svc_ov_un,
         randomforest, randomforest_ov, randomforest_ov_un,
         adaboost, adaboost_ov, adaboost_ov_un,
         xgboost, xgboost_ov, xgboost_ov_un
        ]
plt = plot_boxplot(data2, label_description)
#plt.show()
plt.savefig('matplotlib/data-twitter/training/boxplot/mnb_vs_all.png', format='png')
plt.close()

end = timeit.default_timer()
print ('Duração: %f segundos' % (end - start))