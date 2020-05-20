
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
import lightgbm as lgb

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


# In[3]:


#### Directory
data_path = os.path.dirname(os.path.realpath('lightgbm_train')) + '/'


# In[4]:
print("********* KAGGLE PUBG: LIGHTGBM TRAINING *********")

# compute eigendecomposition of data covariance matrix for PCA transformation
def PCA(x,**kwargs):
    # regularization parameter for numerical stability
    lam = 10**(-7)
    if 'lam' in kwargs:
        lam = kwargs['lam']

    # create the correlation matrix
    P = float(x.shape[1])
    Cov = 1/P*np.dot(x,x.T) + lam*np.eye(x.shape[0])

    # use numpy function to compute eigenvalues / vectors of correlation matrix
    d,V = np.linalg.eigh(Cov)
    return d,V

# PCA-sphereing - use PCA to normalize input features
def PCA_sphereing(x,**kwargs):

    # Step 1: mean-center the data
    x_means = np.mean(x,axis = 1)[:,np.newaxis]
    x_centered = x - x_means

    # Step 2: compute pca transform on mean-centered data
    d,V = PCA(x_centered,**kwargs)

    # Step 3: divide off standard deviation of each (transformed) input,
    # which are equal to the returned eigenvalues in 'd'.
    stds = (d[:,np.newaxis])**(0.5)
    normalizer = lambda data: np.dot(V.T,data - x_means)/stds

    # create inverse normalizer
    inverse_normalizer = lambda data: np.dot(V,data*stds) + x_means

    # return normalizer
    return normalizer,inverse_normalizer

#### Functions that will be used defined here
def feature_engineering(df,is_train=True,debug=True,pca=True,**kwargs):
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

if os.path.isfile(data_path + 'lgb_training_data_pca.pkl'):
    print('* Found previously preprocessed lgb dataset. Loading...')
    with open(data_path + 'lgb_training_data_pca.pkl','rb') as f:
        lgtrain = pickle.load(f)
    with open(data_path + 'lgb_testing_data_pca.pkl','rb') as f:
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


    ## Convert into lgb dataset
    lgtrain = lgb.Dataset(x_train,label = y_train)
    print('* Data converted into lgb dataset')
    save_object(lgtrain, data_path + 'lgb_training_data_pca.pkl')
    save_object(x_test, data_path + 'lgb_testing_data_pca.pkl')

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

params_grid = {"objective": ["regression"],
          "metric": ["mae"],
          "num_iterations": [30000],
          "learning_rate": [0.05],
          "early_stopping_round": [100],
          "num_leaves": [100],
          "boosting": ["gbdt"],
          "bagging_fraction": [0.8],
          "bagging_seed": [0],
          "num_threads": [100],
          "colsample_bytree": [0.7]
         }

### Write a wrapper for lgb_cv, in order to include early stopping feature
def tune_ind_params(params,data,nfold,**kwargs):
    # Credit to julioasotodv at https://github.com/Microsoft/LightGBM/issues/1044
    cv_result = lgb.cv(params,
                         data,
                         nfold=nfold,
                         stratified=False,
                         verbose_eval=1000)
    ## Adding the optimal number of trees chosen by early stopping to the hyperparameter dict
    optimal_num_trees = len(cv_result["l1-mean"])
    params["optimal_number_of_trees"] = optimal_num_trees

    return (params,cv_result["l1-mean"][-1])

def lgb_cv_tuning(grid,data,nfold,return_best=True,parallel=True,**kwargs):
    # Modified implementing parallelism
    print('* Start hyperparameter tuning with {}-fold CV...'.format(nfold))
    print('* Hyperparameter grid:')
    print(params_grid)

    cv_results = []
    all_params = ParameterGrid(grid)
    if parallel:
        num_cores = multiprocessing.cpu_count()
        print("* Parallel mode activated")
        print("* Number of cores detected: ",num_cores)
        print('* Begin CV')
        cv_results = Parallel(n_jobs=num_cores)(delayed(tune_ind_params)(all_params[i],data,nfold) for i in range(len(all_params)))
    else:
        for i in range(all_params[i]):
            print('Hyperparemeter set: {}'.format(i))
            print('* Begin CV')
            cv_results.append(tune_ind_params(params,data,nfold))

    if return_best:
        return min(cv_results,key = lambda x: x[1])
    else:
        return cv_results



### CV
# best_cv = lgb_cv_tuning(params_grid,lgtrain,nfold=10)
best_cv = lgb_cv_tuning(params_grid,lgtrain,nfold=5)
print(best_cv)
best_params = best_cv[0]
best_params["num_iterations"] = best_params["optimal_number_of_trees"]
best_params.pop("optimal_number_of_trees")
best_params.pop("early_stopping_round")
print(best_cv)

save_object(best_cv, data_path + 'lgb_best_cv_112418.pkl')


# In[ ]:


### Train
print('* Training LightGBM model...')
model = lgb.train(best_params,lgtrain,verbose_eval=1000)
save_object(model, data_path + 'lgb_model_112418.pkl')

print('* Making final prediction on testing dataset...')
pred = model.predict(x_test)
save_object(pred, data_path + 'lgb_pred_112418.pkl')
