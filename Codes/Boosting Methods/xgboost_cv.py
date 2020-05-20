
### Loading libraries

# Data manipulation
import numpy as np
import pandas as pd

# Model
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# Plot
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

# System and miscellaneous
import os
import time
import gc, sys
import pickle
from joblib import Parallel, delayed
import multiprocessing
gc.enable()


# In[3]

#### Global handle
is_pca = True
is_return_best = False



#### Directory
data_path = os.path.dirname(os.path.realpath('xgboost_train')) + '/'


# In[4]:
print("********* KAGGLE PUBG: XGBOOST TRAINING *********")

#### Functions that will be used defined here
def feature_engineering(df,is_train=True,debug=True,pca=is_pca,**kwargs):
    if is_train:
        print("* Start feature engineering for training data...")
    else:
        print("* Start feature engineering for testing data...")

    test_idx = None
    if is_train:
        print("* Processing train.csv")
        if debug == True:
            df = df[:10000]

        df = df[df['maxPlace'] > 1]
    else:
        print("* Processing test.csv")
        test_idx = df.Id

    target = 'winPlacePerc'

    print("* Adding new features...")

    df['headshotrate'] = df['kills']/df['headshotKills']
    df['killStreakrate'] = df['killStreaks']/df['kills']
    df['healthitems'] = df['heals'] + df['boosts']
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['distance_over_weapons'] = df['totalDistance'] / df['weaponsAcquired']
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']
    df["skill"] = df["headshotKills"] + df["roadKills"]

    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN

    print("* Removing Na's from df...")
    df.fillna(0, inplace=True)


    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchType")

    y = None

    if is_train:
        print("* Obtain labels from training data")
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)
        features.remove(target)

    print("* Get group mean feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    if is_train: df_out = agg.reset_index()[['matchId','groupId']]
    else: df_out = df[['matchId','groupId']]

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])

    print("* Get group sum feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('sum')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_sum", "_sum_rank"], how='left', on=['matchId', 'groupId'])

    print("* Get group max feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])

    print("* Get group min feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])

    print("* Get group size feature")
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])

    print("* Get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])

    # print("get match type feature")
    # agg = df.groupby(['matchId'])[matchType.columns].agg('mean').reset_index()
    # df_out = df_out.merge(agg, suffixes=["", "_match_type"], how='left', on=['matchId'])

    print("* Get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    df_out = df_out.merge(agg, how='left', on=['matchId'])

    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    feature_names = list(df_out.columns)

    if pca:
        print("* PCA sphereing input feature matrix...")
        df_out_pca = df_out.values.T
        pca_sphere,inverse_sphere = PCA_sphereing(df_out_pca)
        df_out_pca = pca_sphere(df_out_pca).T
        df_out = pd.DataFrame(data=df_out_pca,columns=feature_names)

    X = df_out

    print("* Feature engineering completed!")

    del df, df_out, agg, agg_rank
    gc.collect()

    return X, y, feature_names, test_idx



# Thanks and credited to https://www.kaggle.com/gemartin who created this wonderful mem reducer
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    print('* Reducing memory usage...')
    start_mem = df.memory_usage().sum()
    print('* Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum()
    print('* Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('* Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    print('* Object saved to: {}'.format(filename))


# In[6]:
if is_pca:
    if os.path.isfile(data_path + 'xgb_training_data_pca.pkl'):
        print('* Found previously preprocessed xgb dataset. Loading...')
        with open(data_path + 'xgb_training_data_pca.pkl','rb') as f:
            xgb_train = pickle.load(f)
            x_train = xgb_train[0]
            y_train = xgb_train[1]
            train_columns = xgb_train[2]
            ## Convert into dmatrix
            dtrain = xgb.DMatrix(x_train, label=y_train)
            print('* Data converted into xgb dataset')
        with open(data_path + 'xgb_testing_data_pca.pkl','rb') as f:
            x_test = pickle.load(f)
    else:
        ## Read in data
        print('* Reading in training and testing data from csv, this may take a while...')
        train = pd.read_csv(data_path + '/data/train_V2.csv')
        test = pd.read_csv(data_path + '/data/test_V2.csv')


        # In[ ]:


        ## Feature engineering
        x_train, y_train, train_columns, _ = feature_engineering(train,True,False)
        x_test, _, _ , test_idx = feature_engineering(test,False,True)


        # In[ ]:


        ## Reduce memory
        x_train = reduce_mem_usage(x_train)
        x_test = reduce_mem_usage(x_test)


        # In[ ]:
        save_object([x_train,y_train,train_columns], data_path + 'xgb_training_data_pca.pkl')
        save_object(x_test, data_path + 'xgb_testing_data_pca.pkl')

        ## Convert into dmatrix
        dtrain = xgb.DMatrix(x_train, label=y_train)
        print('* Data converted into xgb dataset')

else:
    if os.path.isfile(data_path + 'xgb_training_data.pkl'):
        print('* Found previously preprocessed xgb dataset. Loading...')
        with open(data_path + 'xgb_training_data.pkl','rb') as f:
            xgb_train = pickle.load(f)
            x_train = xgb_train[0]
            y_train = xgb_train[1]
            train_columns = xgb_train[2]
            ## Convert into dmatrix
            dtrain = xgb.DMatrix(x_train, label=y_train)
            print('* Data converted into xgb dataset')
        with open(data_path + 'xgb_testing_data.pkl','rb') as f:
            x_test = pickle.load(f)
    else:
        ## Read in data
        print('* Reading in training and testing data from csv, this may take a while...')
        train = pd.read_csv(data_path + '/data/train_V2.csv')
        test = pd.read_csv(data_path + '/data/test_V2.csv')


        # In[ ]:


        ## Feature engineering
        x_train, y_train, train_columns, _ = feature_engineering(train,True,False)
        x_test, _, _ , test_idx = feature_engineering(test,False,True)


        # In[ ]:


        ## Reduce memory
        x_train = reduce_mem_usage(x_train)
        x_test = reduce_mem_usage(x_test)


        # In[ ]:
        save_object([x_train,y_train,train_columns], data_path + 'xgb_training_data.pkl')
        save_object(x_test, data_path + 'xgb_testing_data.pkl')

        ## Convert into dmatrix
        dtrain = xgb.DMatrix(x_train, label=y_train)
        print('* Data converted into xgb dataset')


# In[20]:


## Hyperparameter tuning with 10-fold CV for 10 iterations
### Define a parameter grid
# params_grid = {"objective": ["regression"],
#           "metric": ["mae"],
#           "num_iterations": [20000,30000],
#           "learning_rate": [0.1,0.05,0.01],
#           "early_stopping_round": [200],
#           "num_leaves": [31,40,60],
#           "boosting": ["gbdt","dart"],
#           "bagging_fraction": [0.7],
#           "bagging_seed": [0],
#           "num_threads": [60],
#           "colsample_bytree": [0.7]
#          }

# params_grid = {"booster": ['gbtree'],
#           "objective": ['reg:linear'],
#           "eta": [0.01,0.05],
#           "max_depth": [0],
#           "subsample": [0.7,0.8],
#           "seed": [0],
#           "colsample_bytree": [0.7,0.8]
#          }

params_grid = {"booster": ['gbtree'],
          "objective": ['reg:linear'],
          "eta": [0.1],
          "max_depth": [10],
          "subsample": [0.8],
          "seed": [0],
          "colsample_bytree": [0.8]
         }

### Write a wrapper for xgb_cv, in order to include early stopping feature
def tune_ind_params(params,data,nfold,niter,earlystopping,return_best=is_return_best,**kwargs):
    # Credit to julioasotodv at https://github.com/Microsoft/LightGBM/issues/1044
    cv_result = xgb.cv(params=params,
                         dtrain=data,
                         nfold=nfold,
                         metrics=['mae'],
                         num_boost_round=niter,
                         early_stopping_rounds=earlystopping,
                         stratified=False,
                         verbose_eval=1000,
                         as_pandas = True)
    ## Adding the optimal number of trees chosen by early stopping to the hyperparameter dict
    optimal_num_trees = len(cv_result["test-mae-mean"])
    if optimal_num_trees != niter:
        print("* Early stopping at {} boosting rounds".format(optimal_num_trees))
        params["optimal_number_of_trees"] = optimal_num_trees
    else:
        print("* Full {} boosting rounds finished".format(niter))

    # print(cv_result)
    print(cv_result.iloc[-1])
    if return_best:
        return (params,cv_result.iloc[-1]["test-mae-mean"])
    else:
        return (params,cv_result)

def xgb_cv_tuning(grid,data,nfold,niter,earlystopping,return_best=is_return_best,parallel=False,**kwargs):
    # Modified implementing parallelism
    print('* Start hyperparameter tuning with {}-fold CV...'.format(nfold))
    print('* Hyperparameter grid:')
    print(params_grid)

    cv_results = []
    all_params = ParameterGrid(grid)
    if parallel:
        # num_cores = multiprocessing.cpu_count()
        print("* Parallel mode activated")
        print("* Number of cores detected: ",50)
        print('* Begin CV')
        cv_results = Parallel(n_jobs=50)(delayed(tune_ind_params)(all_params[i],data,nfold,niter,earlystopping) for i in range(len(all_params)))
    else:
        for i in range(len(all_params)):
            print('Hyperparemeter set: {}'.format(i))
            print('* Begin CV')
            cv_results.append(tune_ind_params(all_params[i],data,nfold,niter,earlystopping))

    if return_best:
        return min(cv_results,key = lambda x: x[1])
    else:
        return cv_results



### CV
if is_return_best:
    best_cv = xgb_cv_tuning(params_grid,dtrain,nfold=5,niter=30000,earlystopping=100)
    print(best_cv)
    best_params = best_cv[0]
    nround = best_params["optimal_number_of_trees"]
    best_params.pop("optimal_number_of_trees")
    print("* Best rounds = ",nround)
    save_object(best_params, data_path + 'xgb_pca_best_params_01_10_30000.pkl')
else:
    cv = xgb_cv_tuning(params_grid,dtrain,nfold=5,niter=30000,earlystopping=100)
    save_object(cv, data_path + 'xgb_cv_history_pca_01_10_30000.pkl')



# best_params = {"booster": 'gbtree',
#           "objective": 'reg:linear',
#           "eta": 0.05,
#           "max_depth": 15,
#           "subsample": 0.8,
#           "seed": 0,
#           "colsample_bytree": 0.8,
#           "optimal_number_of_trees": 30000
#          }

# In[ ]:


### Train
# print('* Training XGBoost model...')
# xgbmodel = xgb.train(best_params,dtrain,nround,verbose_eval=1000)
# xgbmodel.save_model(data_path + 'xgb_model_005_15_30000.model')
#
# print('* Making final prediction on testing dataset...')
# dtest = xgb.DMatrix(x_test)
# xgbpred = xgbmodel.predict(dtest)
# save_object(xgbpred, data_path + 'xgb_pred_005_15_30000.pkl')
