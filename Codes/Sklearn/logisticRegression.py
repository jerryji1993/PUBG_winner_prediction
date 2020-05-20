#0.1778 for 0.5
from autograd import numpy as np
import pandas as pd
from sklearn import preprocessing
from autograd import grad
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import gc, sys
import time
gc.enable()

INPUT_DIR = "./input/"

def feature_engineering(is_train=True):

    if is_train:
        print("processing train.csv")
        #df = pd.read_csv(INPUT_DIR + 'train_V2.csv', nrows = 10000)
        df = pd.read_csv(INPUT_DIR + 'train_V2.csv')
        df = df[df['maxPlace'] > 1]
    else:
        print("processing test.csv")
        #df = pd.read_csv(INPUT_DIR + 'test_V2.csv', nrows = 10000)
        df = pd.read_csv(INPUT_DIR + 'test_V2.csv')
    # df = reduce_mem_usage(df)
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]



    print("remove some columns")
    target = 'winPlacePerc'
    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")

    features.remove("matchType")


    y = None

    print("get target")
    if is_train:
        y = np.array(df.groupby(['matchId', 'groupId'])[target].agg('mean'), dtype=np.float64)
        features.remove(target)

    print("get group mean feature")
    agg = df.groupby(['matchId', 'groupId'])[features].agg('mean')
    #agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    if is_train:
        df_out = agg.reset_index()[['matchId', 'groupId']]
    else:
        df_out = df[['matchId', 'groupId']]

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])


    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    X = np.array(df_out, dtype=np.float64)

    feature_names = list(df_out.columns)

    del df, df_out, agg
    gc.collect()

    return X, y, feature_names


data_x, y, feature_names = feature_engineering(True)
w0 = 0.1* np.random.rand(data_x.shape[1]+ 1, 1)

x_means = np.mean(data_x,axis = 0)[np.newaxis,:]
x_stds = np.std(data_x,axis = 0)[np.newaxis,:]
x = (data_x - x_means)/(x_stds+0.0000001)

print (x.shape)
print (y.shape)
# y = 2 * y - 1
for i in range(len(y)):
	if y[i] < 0.5:
		y[i] = -1
	else:
		y[i] = 1

reg = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000, multi_class='auto').fit(x, y)

print (reg.score(x, y))



x_test, _, _ = feature_engineering(False)
x_means = np.mean(x_test,axis = 0)[np.newaxis,:]
x_stds = np.std(x_test,axis = 0)[np.newaxis,:]
x_test = (x_test - x_means)/(x_stds+0.0000001)
print(x_test.shape)

y_predict =  reg.predict_proba(x_test)
y_predict = y_predict.reshape(-1,2)

#yy = reg.predict(x_test)
#yy = yy.reshape(-1,1)
#print (yy[:100])
#print (y_predict[:100][:])
#y_predict = (y_predict + 1) / 2

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y_predict = y_predict[:,1]
print(y_predict.shape)
y_predict = y_predict.reshape(-1,1)
scaler.fit(y_predict)
y_predict = scaler.transform(y_predict)

df_test = pd.read_csv(INPUT_DIR + 'test_V2.csv')

df_test['winPlacePerc'] = y_predict

for i in range(len(df_test)):
    winPlacePerc = y_predict[i][0]
    maxPlace = int(df_test.iloc[i]['maxPlace'])
    if maxPlace == 0:
        winPlacePerc = 0.0

    elif maxPlace == 1:
        winPlacePerc = 1.0
    else:
        gap = 1.0 / (maxPlace - 1)
        winPlacePerc = round(winPlacePerc / gap) * gap

    if winPlacePerc < 0: winPlacePerc = 0.0
    if winPlacePerc > 1: winPlacePerc = 1.0
    y_predict[i][0] = winPlacePerc



df_test['winPlacePerc'] = y_predict
submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('logisticRegression_1.csv', index=False)

