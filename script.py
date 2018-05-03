"""
A non-blending lightGBM model that incorporates portions and ideas from various public kernels.
"""

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os
import lib.data_preprocessing as pp
import lib.lightGBM_fitting as lgbmf
from threading import Thread

### Global vars
debug=1
nrows=184903891-1
nchunk=25000000
val_size=2500000

frm=nrows-75000000
if debug:
    frm=0
    nchunk=100000
    val_size=10000
to=frm+nchunk
# directories
test_filedir = os.getcwd() + "\\input\\test.csv"
# test_filedir = os.getcwd() + "/input/test.csv" # For UNIX
train_filedir = os.getcwd() + "\\input\\train.csv"
# train_filedir = os.getcwd() + "/input/train.csv" # For UNIX
### Global vars

if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')

# For Yuanze
def DO(frm,to,fileno):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            } # Define the data type of the data got loaded
    test_usecols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']
    train_usecols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']

    print('loading training data...',frm,to)
    train_df = pd.read_csv(train_filedir, parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=train_usecols)

    print('loading testing data...')
    if debug:
        test_df = pd.read_csv(test_filedir, nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=test_usecols)
    else:
        test_df = pd.read_csv(test_filedir, parse_dates=['click_time'], dtype=dtypes, usecols=test_usecols)

    len_train = len(train_df)
    train_df=train_df.append(test_df)
    # Train_df is a Panda dataframe
    
    del test_df
    gc.collect()
    
    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    train_df['wday'] = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8') # Which day of the week
    
    gc.collect()

    train_df = pp.do_preprocessing(train_df)

    train_df, predictors = pp.do_prev_Click(train_df, predictors=[])

    train_df, predictors = pp.do_generating_nextClick(train_df, frm, to, debug, predictors)


    print("vars and data type: ")
    train_df.info()
    train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

    target = 'is_attributed'
    predictors.extend(['app','device','os', 'channel', 'hour', 'day', 
                  'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                  'ip_app_os_count', 'ip_app_os_var',
                  'ip_app_channel_var_day','ip_app_channel_mean_hour',
                  'ip_channel_countuniq','ip_dev_os_app_cumcount', 
                  'ip_day_hour_countuniq', 'ip_app_countuniq', 'ip_app_os_countuniq', 
                  'ip_dev_countniq', 'app_channel_countuniq', 'ip_os_count', 
                  'ip_dev_os_countuniq', 'ip_app_channel_var', 'app_os_channel_countuniq',
                  'app_os_channel_var','app_os_wday_var'])
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
    print('predictors',predictors)
    # 'X1', 

    test_df = train_df[len_train:]
    val_df = train_df[(len_train-val_size):len_train]
    train_df = train_df[:(len_train-val_size)]

    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    gc.collect()

    print("Training...")
    start_time = time.time()
    
    (bst,best_iteration) = lgbmf.lgb_modelfit_nocv( 
                            train_df, 
                            val_df, 
                            predictors, 
                            target, 
                            objective='binary', 
                            metrics='auc',
                            early_stopping_rounds=30, 
                            verbose_eval=True, 
                            num_boost_round=1000, 
                            categorical_features=categorical)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()


    

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
    if not debug:
        print("writing...")
        sub.to_csv('prediction_%d.csv'%(fileno),index=False,float_format='%.9f')
    print("done...")

    print('Plot feature importances...')

    ax = lgb.plot_importance(bst, max_num_features=100)
    plt.show()

    return sub
# For Yuanze 

# Run the main function 
sub=DO(frm,to,0)
