# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl    
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import glob 
pl.rc('text', usetex=False)
pl.rc('font',**{'family':'serif','serif':['Palatino']})

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
    #%%
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
    
    #%%
    data.dropna(inplace=True)
    data.index=[pd.to_datetime(i.split(' 00')[0]) for i in data['Timestamp as DateTime']]
    data.drop_duplicates()
    
    node_id=1
    data=data[data['Node Id']==node_id]
    

    cols =['Temperature','Conductivity', 'pH',]
    cols+=['DissolvedOxygen']
    for c in cols:
        data=data[data[c]>0]
    
    X=data[cols]
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
    #mask = np.triu(np.ones_like(corr, dtype=np.bool))
    mask = np.triu(np.ones_like(corr))
    heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    #heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=0);
    pl.savefig(ds_name+'_correlation.png',  bbox_inches='tight', dpi=300)
    pl.show()

    # #pl.figure(figsize=(6, 6))
    # g = sns.pairplot(X, diag_kind="kde", corner=False)
    # g.map_lower(sns.kdeplot, levels=4, color=".2")
    # g.map_upper(sns.regplot, )
    # pl.show()
    
    # #pl.figure(figsize=(6, 6))
    # g = sns.pairplot(X_train, )
    # g.map_diag(sns.distplot)
    # g.map_lower(sns.regplot)
    # g.map_upper(r2_coef)
    # pl.show()
    
    
    #X_train, y_train = B_train[variable_names].values, B_train[target_names].values, 
    #X_test , y_test  = B_test [variable_names].values, B_test [target_names].values, 
                       
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
      'normalize'       : None,
      }
    #%%
    return regression_data
#%%
