print('HELLO FROM CONTAINER LUIS')

import pandas as pd
import os
#from git import Repo
from  loguru import logger
import os

if not os.path.exists('./data/'):
    logger.info('creatingi dir data')
    os.makedirs('./data')
else:
    logger.info('path already created')

df =  pd.DataFrame()
df.to_csv('./data/p1.csv')
logger.info('data saved')

#logger.info('clonning git repo')

#Repo.clone_from('https://github.com/DataTalksClub/mlops-zoomcamp.git', '/app/data/main_repo/')
#logger.info('repo clonned')



logger.info('downloading data')
os.system('')
os.system('wget --no-check-certificate --directory-prefix=/app/data  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet')
os.system('wget --no-check-certificate --directory-prefix=/app/data  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet')
logger.info('downloaded..ok')


logger.info('INIT HOMEWORK-1')

import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

logger.info('reading data')
df = pd.read_parquet('./data/yellow_tripdata_2023-01.parquet')

logger.info(f'Data has {df.shape[1]} columns')

df.tpep_dropoff_datetime  = pd.to_datetime(df.tpep_dropoff_datetime)
df.tpep_pickup_datetime   = pd.to_datetime(df.tpep_pickup_datetime)

df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

std_value = df['duration'].std()
logger.info(f'The Std of the duration data is: {std_value} ')

df2 = df[(df.duration >= 1) & (df.duration <= 60)]

pct_remain = 100*df2.shape[0]/df.shape[0]

logger.info(f'Percentage after filtering data: {pct_remain}')


categorical = ['PULocationID', 'DOLocationID']
df2[categorical] = df2[categorical].astype(str)
train_dicts = df2[categorical].to_dict(orient='records')
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
logger.info(f'Number of columns of X_train: {X_train.shape[1]}')

target = 'duration'
y_train = df2[target].values

logger.info('fitting linear reg model and computing rmse on train set')
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_train)
rmse_train = root_mean_squared_error(y_train, y_pred)

logger.info(f'RMSE-Train: {rmse_train}')

def read_dataframe(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
        df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
        df.tpep_pickup_datetime  = pd.to_datetime(df.tpep_pickup_datetime)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    #loguru.info(df.duration.describe())
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    #print(df.duration.describe())
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df

logger.info('READING TRAIN AND VAL DATA ')
df_train = read_dataframe('./data/yellow_tripdata_2023-01.parquet')
df_val = read_dataframe('./data/yellow_tripdata_2023-02.parquet')


categorical = [ 'PULocationID', 'DOLocationID'] #'PU_DO'] #'PULocationID', 'DOLocationID'
dv = DictVectorizer()
train_dicts = df_train[categorical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)
val_dicts = df_val[categorical].to_dict(orient='records')
X_val = dv.transform(val_dicts)


target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values

logger.info('fitting a linear reg model and computing rmse on val-set')
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

rmse_val = root_mean_squared_error(y_val, y_pred)

logger.info(f'RMSE-val: {rmse_val}')
logger.info('END')
