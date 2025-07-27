#!/usr/bin/python
# -*- coding: utf-8 -*-

# pip install --break-system-packages --user

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from hydroeval import kge
import os
import warnings
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
DATASET_PATH = './data/data_kosovo/data_kosovo.csv'
BASENAME = 'skopt_50_'
N_SPLITS = 3
SCORING = 'neg_root_mean_squared_error'

# Helper Functions
def accuracy_log(y_true, y_pred):
    """Calculate the percentage of predictions within a log10 error threshold."""
    y_true = np.abs(np.array(y_true))
    y_pred = np.abs(np.array(y_pred))
    return (np.abs(np.log10(y_true / y_pred)) < 0.3).sum() / len(y_true) * 100


def read_kosovo(seed=None):
    #%%
    ds_name='WQ-Kosovo'
    fn='./data/data_kosovo/data_kosovo.csv'
            
    data=pd.read_csv(fn,sep=';')
    thresh=18360
    
    plt.figure()
    plt.plot(list(data['DissolvedOxygen'][:thresh])+[None for i in range(thresh,len(data))],'b-',label='Data used')
    plt.plot([None for i in range(thresh)]+list(data['DissolvedOxygen'][thresh:]),'r-',label='Data discarded',)
    plt.legend()
    plt.savefig(ds_name+'_ts.png',  bbox_inches='tight', dpi=300)
    plt.show()
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
    
    plt.figure(figsize=(5, 5))
    corr = df.corr()
    #mask = np.triu(np.ones_like(corr, dtype=np.bool))
    mask = np.triu(np.ones_like(corr))
    heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    #heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=0);
    plt.savefig(ds_name+'_correlation.png',  bbox_inches='tight', dpi=300)
    plt.show()

    # #plt.figure(figsize=(6, 6))
    # g = sns.pairplot(X, diag_kind="kde", corner=False)
    # g.map_lower(sns.kdeplot, levels=4, color=".2")
    # g.map_upper(sns.regplot, )
    # plt.show()
    
    # #plt.figure(figsize=(6, 6))
    # g = sns.pairplot(X_train, )
    # g.map_diag(sns.distplot)
    # g.map_lower(sns.regplot)
    # g.map_upper(r2_coef)
    # plt.show()
    
    
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


def run_experiment(datasets):
    """Run experiments using genetic algorithm-based hyperparameter optimization."""
    n_runs = len(datasets)
    for run in range(n_runs):
        random_seed = run + 37
        for dataset in datasets:
            dr = dataset['name'].replace(' ', '_').replace("'", "").lower()
            path = f'./json_minmax_50_2_{dr}/'
            os.makedirs(path, exist_ok=True)

            for tk, tn in enumerate(dataset['target_names']):
                target = dataset['target_names'][tk]
                y_train, y_test = dataset['y_train'][tk], dataset['y_test'][tk]
                X_train, X_test = dataset['X_train'], dataset['X_test']
                n_samples_train, n_features = dataset['n_samples'], dataset['n_features']
                task, normalize = dataset['task'], dataset['normalize']

                # Print dataset information
                print("=" * 80)
                print(f"Dataset: {dataset['name']} -- {target}")
                print(f"Training Samples: {n_samples_train}")
                print(f"Testing Samples: {len(y_test)}")
                print(f"Features: {n_features}")
                print(f"Normalization: {normalize}")
                print(f"Task: {task}")
                print(f"Reference: {dataset['reference']}")
                print("=" * 80)

               
                # Define models and parameter grids
                estimators = [
                    ('EN', ElasticNet(random_state=random_seed), {
                        "alpha": Continuous(0.001, 100),
                        "l1_ratio": Continuous(0, 1),
                        'fit_intercept': Categorical([True, False]),
                        'positive': Categorical([True, False]),
                    }),
                    ('SVR', SVR(kernel='rbf', max_iter=500), {
                        "gamma": Continuous(0.001, 10),
                        "epsilon": Continuous(0.001, 100),
                        "C": Continuous(0.001, 1e5),
                    }),
                    ('LGBM', LGBMRegressor(random_state=random_seed), {
                        "n_estimators": Integer(5, 300),
                        'boosting_type': Categorical(['gbdt', 'dart', 'goss']),
                        "max_depth": Integer(2, 10),
                        "learning_rate": Continuous(0.001, 0.1, distribution="log-uniform"),
                        "reg_alpha": Continuous(0.001, 10),
                        "reg_lambda": Continuous(0.001, 10),
                    }),
                ]

                # Run genetic algorithm optimization
                cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_seed)
                for est_name, estimator, param in estimators:
                    est_gp = GASearchCV(
                        estimator=estimator,
                        param_grid=param,
                        cv=cv,
                        scoring=SCORING,
                        population_size=50,
                        generations=30,
                        tournament_size=3,
                        elitism=True,
                        crossover_probability=0.9,
                        mutation_probability=0.08,
                        criteria="max",
                        algorithm="eaMuCommaLambda",
                        error_score='raise',
                        n_jobs=-1
                    )

                    est_gp.fit(X_train, y_train)
                    y_test_pred = est_gp.predict(X_test).ravel()

                    # Evaluate metrics
                    r2 = r2_score(y_test, y_test_pred)
                    r = pearsonr(y_test, y_test_pred)[0]
                    mse = mean_squared_error(y_test, y_test_pred)
                    acc = accuracy_log(y_test, y_test_pred)
                    kge_ = kge(y_test, y_test_pred)[0][0]

                    # Print results
                    print("-" * 80)
                    print(f"R2Score: {r2}")
                    print(f"Rscore: {r}")
                    print(f"MSE: {mse}")
                    print(f"ACC: {acc}")
                    print(f"KGE: {kge_}")
                    print("-" * 80)

                    # Save results to JSON
                    sim = {
                        'Y_TEST_TRUE': y_test,
                        'Y_TEST_PRED': y_test_pred,
                        'EST_NAME': est_name,
                        'ALGO': est_name,
                        'EST_PARAMS': est_gp.best_estimator_.get_params(),
                        'OPT_PARAMS1': est_gp.best_params_,
                        'OUTPUT': target,
                        'TARGET': target,
                        'SEED': random_seed,
                        'ACTIVE_VAR_NAMES': dataset['feature_names'],
                        'ACTIVE_VAR': dataset['feature_names'],
                        'SCALER': None,
                    }

                    ds_name = dataset['name'].replace('/', '_').replace("'", "").lower()
                    tg_name = target.replace('/', '_').replace("'", "").lower()
                    algo = sim['ALGO'].split(':')[0]
                    json_file = (
                        f"{path}{BASENAME}_minmax_run_{run:02d}_{ds_name}_{sim['EST_NAME']}_{algo}_{tg_name}.json"
                    ).replace(' ', '_').replace("'", "").lower()

                    pd.DataFrame([sim]).to_json(json_file)
                    
                    plt.figure(figsize=(5, 5))
                    plt.plot(y_test, y_test, 'r-', y_test, y_test_pred, 'b.', label='Predicted')
                    plt.title(f"{dataset['name']} - {target} - {est_name}")
                    plt.legend()
                    plt.savefig(f"{path}{BASENAME}_{run:02d}_{est_name}_{target}.png", bbox_inches='tight')
                    plt.show()

if __name__ == "__main__":
    datasets = [read_kosovo(seed=run + 37) for run in range(50)]
    run_experiment(datasets)

