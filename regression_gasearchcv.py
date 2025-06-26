
#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pylab as pl
import os, math
import sys, getopt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import graphviz

import matplotlib.pyplot as plt
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space
from sklearn_genetic.callbacks import LogbookSaver, ProgressBar
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
#%%

program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)

#print ("This is the name of the script: ", program_name)
#print ("Number of arguments: ", len(arguments))
#print ("The arguments are: " , arguments)

if len(arguments)>0:
    if arguments[0]=='-r':
        run0 = int(arguments[1])
        n_runs = run0+1
else:
    run0, n_runs = 0, 1


#%%----------------------------------------------------------------------------

def accuracy_log(y_true, y_pred):
    y_true = np.abs(np.array(y_true))
    y_pred = np.abs(np.array(y_pred))

    return (np.abs(np.log10(y_true / y_pred)) < 0.3).sum() / len(y_true) * 100

# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import glob
from sklearn.preprocessing import MinMaxScaler
pl.rc('text', usetex=False)
#pl.rc('font',**{'family':'serif','serif':['Palatino']})

from scipy.stats import pearsonr
def r_coef(x,y,label=None,color=None,**kwargs):
    ax = pl.gca()
    r,p = pearsonr(x,y)
    ax.annotate('r = {:.2f}'.format(r), xy=(0.5,0.5), xycoords='axes fraction', ha='center')
    ax.set_axis_off()

def r2_coef(x,y,label=None,color=None,**kwargs):
    ax = pl.gca()
    r = r2_score(x,y)
    ax.annotate('r = {:.2f}'.format(r), xy=(0.5,0.5), xycoords='axes fraction', ha='center')
    ax.set_axis_off()


#%%
def read_kosovo(seed=None):

    ds_name='WQ-Kosovo'
    fn='./data/data_kosovo/data_kosovo.csv'

    data=pd.read_csv(fn,sep=';')
    thresh=18360

    pl.figure()
    pl.plot(list(data['DissolvedOxygen'][:thresh])+[None for i in range(thresh,len(data))],'b-',label='Data used')
    pl.plot([None for i in range(thresh)]+list(data['DissolvedOxygen'][thresh:]),'r-',label='Data discarded',)
    pl.legend()
    pl.savefig(ds_name+'_ts.png',  bbox_inches='tight', dpi=300)
    pl.show()
    data=data.iloc[:thresh]


    data.dropna(inplace=True)
    data.index=[pd.to_datetime(i.split(' 00')[0]) for i in data['Timestamp as DateTime']]
    data.drop_duplicates()

    node_id=1
    data=data[data['Node Id']==node_id]


    cols =['Temperature','Conductivity', 'pH',]
    cols+=['DissolvedOxygen']
    for c in cols:
        data=data[data[c]>0]

    #Normalizar
    # Inicializar o objeto MinMaxScaler
    scaler = MinMaxScaler()
    # Selecionar apenas as colunas da 3Âª atÃ© a penÃºltima
    columns_to_normalize = data.columns[4:]
    df_normalized = data.copy()  # Crie uma cÃ³pia do DataFrame para manter o original intacto
    df_normalized[columns_to_normalize] = scaler.fit_transform(df_normalized[columns_to_normalize])


    X=df_normalized[cols]
    X.dropna(inplace=True)
    X.drop_duplicates(inplace=True)


    X.columns=['T', 'C', 'pH', 'DO']

    target_names=['DO']

    variable_names = list(X.columns.drop(target_names))

    X.describe().to_latex(buf=ds_name.lower()+'-describe.tex', caption='To in inserted', label='tab:wq-desc')

    df = X[variable_names+target_names].copy()
    #sns.set_context("paper")

    pl.figure(figsize=(5, 5))
    corr = df.corr()

    mask = np.triu(np.ones_like(corr))
    heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=0);
    pl.savefig(ds_name+'_correlation.png',  bbox_inches='tight', dpi=300)
    pl.show()


    X_train, X_test, y_train, y_test = train_test_split(
                      X[variable_names].values, X[target_names].values,
                      test_size=0.3, random_state=seed)

    y_train=y_train.T
    y_test=y_test.T


    n=len(y_train);
    n_samples, n_features = X_train.shape

    regression_data =  {
      'task'            : 'regression',
      'name'            : ds_name,
      'feature_names'   : np.array(variable_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples,
      'n_features'      : n_features,
      'X_train'         : X_train,
      'y_train'         : y_train,
      'X_test'          : X_test,
      'y_test'          : y_test,
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'reference'       : "https://doi.org/10.1016/j.dib.2022.108486",
      'items'           : None,
      'normalize'       : 'yes',
      }

    return regression_data

basename='skopt_50_'

#from kosovo_read_data import *

pd.options.display.float_format = '{:.3f}'.format
n_splits    = 3
scoring     = 'neg_root_mean_squared_error'
for run in range(50):
    random_seed=run+37
    datasets = [
                read_kosovo(seed=random_seed),
                #read_wollongong(seed=random_seed)
               ]

    for dataset in datasets:#[:1]:
        dr=dataset['name'].replace(' ','_').replace("'","").lower()
        path='./json_minmax_50_2_'+dr+'/'
        os.system('mkdir  '+path)


        for tk, tn in enumerate(dataset['target_names']):
            print (tk, tn)
            target                          = dataset['target_names'][tk]
            y_train_, y_test_               = dataset['y_train'][tk], dataset['y_test'][tk]
            dataset_name, X_train, X_test   = dataset['name'], dataset['X_train'], dataset['X_test']
            n_samples_train, n_features     = dataset['n_samples'], dataset['n_features']
            task, normalize                 = dataset['task'], dataset['normalize']
            y_train                         = y_train_
            y_test                          = y_test_
            n_samples_test                  = len(y_test)
            np.random.seed(random_seed)

            s=''+'\n'
            s+='='*80+'\n'
            s+='Dataset                    : '+dataset_name+' -- '+target+'\n'
            s+='Number of training samples : '+str(n_samples_train) +'\n'
            s+='Number of testing  samples : '+str(n_samples_test) +'\n'
            s+='Number of features         : '+str(n_features)+'\n'
            s+='Normalization              : '+str(normalize)+'\n'
            s+='Task                       : '+str(dataset['task'])+'\n'
            s+='Reference                  : '+str(dataset['reference'])+'\n'
            s+='='*80
            s+='\n'

            minmax=MinMaxScaler()
            X_train = minmax.fit_transform(X_train)
        
            y_train = minmax.fit_transform(y_train.reshape(-1,1))
            
            print(s)
            #args = (X_train, y_train, X_test, y_test, 'eval', task,  n_splits,
                    #int(random_seed), scoring, target,
                    #n_samples_train, n_samples_test, n_features)

            feature_names=dataset['feature_names']

            cv = KFold(n_splits=n_splits, shuffle=True,random_state=random_seed)

            estimators=[
                    (
                    'EN',
                    ElasticNet(random_state=random_seed),
                    {
                        "alpha": Continuous(0.001, 100,),
                        "l1_ratio": Continuous(0, 1,),
                        'fit_intercept':Categorical([True, False]),
                        'positive':Categorical([True, False]),
                    }
                    ),
                    (
                    'SVR',
                    SVR(kernel='rbf',max_iter=500),
                    {
                    #    'kernel':Categorical(['linear', 'rbf', 'sigmoid',]),
                    #    "degree": Integer(2, 10),
                        "gamma": Continuous(0.001, 10),
                        "epsilon": Continuous(0.001, 100),
                        "C": Continuous(0.001, 1e5),
                    }
                    ),
                    (
                    'LGBM',
                    LGBMRegressor(random_state=random_seed),
                    {
                        "n_estimators": Integer(5, 300),
                        'boosting_type':Categorical(['gbdt','dart','goss',]),
                        #"rg__loss": Categorical(["absolute_error", "squared_error"]),
                        "max_depth": Integer(2, 10),
                        "learning_rate": Continuous(0.001, 0.1, distribution="log-uniform"),
                        "reg_alpha": Continuous(0.001, 10),
                        "reg_lambda": Continuous(0.001, 10),
                    }
                    ),
                ]

            for est_name,estimator, param in estimators:
                list_results=[]
                list_results1=[]
                est_gp = GASearchCV(
                    estimator=estimator,
                    param_grid=param,
                    cv=cv,
                    scoring="neg_root_mean_squared_error",
                    population_size=50,
                    generations=30,
                    tournament_size=3,
                    elitism=True,
                    #keep_top_k=4,
                    crossover_probability=0.9,
                    mutation_probability=0.08,
                    criteria="max",
                    algorithm="eaMuCommaLambda",
                    error_score='raise',
                    n_jobs=-1)
      
                from hydroeval import  kge

                #%%
          
                est_gp.fit(X_train, y_train)
                y_test_pred = est_gp.predict(X_test).ravel()
              

                sim1={'Y_TEST_TRUE'      :y_test,
                         'Y_TEST_PRED'      :y_test_pred,
                         'EST_NAME'         :est_name,
                         'ALGO'             :est_name,
                         'EST_PARAMS'       :est_gp.best_estimator_.get_params(),
                         'OPT_PARAMS1'       :est_gp.best_params_,
                         'OUTPUT'           :target,
                         'TARGET'           :target,
                         'SEED'             :random_seed,
                         'ACTIVE_VAR_NAMES' :dataset['feature_names'],
                         'ACTIVE_VAR'       :dataset['feature_names'],
                         'SCALER'           :None,
                         }

                r2   = r2_score(sim1['Y_TEST_TRUE'].ravel(), sim1['Y_TEST_PRED'].ravel())
                r    = pearsonr(sim1['Y_TEST_TRUE'].ravel(), sim1['Y_TEST_PRED'].ravel())[0]
                mse = mean_squared_error(sim1['Y_TEST_TRUE'].ravel(), sim1['Y_TEST_PRED'].ravel())
                #rmsl = rms(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())
                #mape = MAPE(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())
                acc  = accuracy_log(sim1['Y_TEST_TRUE'].ravel(), sim1['Y_TEST_PRED'].ravel())
                kge_ = kge(sim1['Y_TEST_TRUE'].ravel(), sim1['Y_TEST_PRED'].ravel())[0][0]
                #nse_ = nse(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())

                scores = '-'*80+'\n'
                scores += f"R2Score : {r2}" + '\n'
                scores += f"Rscore  : {r}" + '\n'
                scores += f"MSE    : {mse}" + '\n'
                #scores += f"RMSL    : {rmsl}" + '\n'
                #scores += f"MAPE    : {mape}" + '\n'
                scores += f"ACC     : {acc}"  + '\n'
                scores += f"KGE     : {kge_}" + '\n'
                #scores += f"NSE     : {nse_}" + '\n'
                scores += '-'*80+'\n'

                print(scores)


                if task == 'forecast' or task == 'regression':
                      pl.figure(figsize=(12,5));

                      s = range(len(y_test))

                      pl.plot(sim1['Y_TEST_TRUE'][s].ravel(), 'r-o', label='Real data',)

                      pl.plot(sim1['Y_TEST_PRED'][s].ravel(), 'b-o', label='Predicted',)

                      acc = accuracy_log(sim1['Y_TEST_TRUE'].ravel(), sim1['Y_TEST_PRED'].ravel())

                      pl.title(dataset_name
                                 + ' -- '
                                 + target
                                 + '\nMSE = '
                                 + str(mse)
                                 + ', '
                                 + 'R$^2$ = '
                                 + str(r2)
                                 + ', '
                                 + 'R = '
                                 + str(r)
                                 + 'KGE = '
                                 + str(kge_))

                      pl.ylabel(dataset_name)

                      pl.title(sim1['EST_NAME']
                                 + ': (Testing) R$^2$='
                                 + str('%1.3f' % r2)
                                 + '\t MSE='
                                 + str('%1.3f' % mse)
                                 + '\t R ='
                                 + str('%1.3f' % r)
                                 + '\t KGE ='
                                 + str('%1.3f' % kge_)
                                  )
                      pl.show()

                sim['RUN']          = run;
                sim['DATASET_NAME'] = dataset_name;

                list_results1.append(sim1)
            
                data    = pd.DataFrame(list_results1)
                ds_name = dataset_name.replace('/','_').replace("'","").lower()
                tg_name = target.replace('/','_').replace("'","").lower()
                algo    = sim1['ALGO'].split(':')[0]

                js = (path+ basename+ '_minmax_'+ '_run_'+ str("{:02d}".format(run))+ '_'+ ("%15s"%ds_name         ).rjust(15).replace(' ','_')+ ("%9s"%sim1['EST_NAME']  ).rjust( 9).replace(' ','_')+ ("%10s"%algo            ).rjust(10).replace(' ','_')+ ("%15s"%tg_name         ).rjust(25).replace(' ','_')+ '.json')

                js = js.replace(' ','_').replace("'","").lower()
                js = js.replace('(','_').replace(")","_").lower()
                js = js.replace('[','_').replace("]","_").lower()
                js = js.replace('-','_').replace("_","_").lower()

                data.to_json(js)
#%%
