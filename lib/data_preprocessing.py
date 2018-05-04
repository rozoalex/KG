import gc
import os
import numpy as np
import pandas as pd

# For Kaiyi 
def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=True, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_countuniq( df, group_cols, counted, agg_name, agg_type='uint32', show_max=True, show_agg=True ):
    if show_agg:
        print( "Counting unqiue ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )
    
def do_cumcount( df, group_cols, counted, agg_name, agg_type='uint32', show_max=True, show_agg=True ):
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_mean( df, group_cols, counted, agg_name, agg_type='float32', show_max=True, show_agg=True ):
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_var( df, group_cols, counted, agg_name, agg_type='float32', show_max=True, show_agg=True ):
    if show_agg:
        print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )
# For Kaiyi

def do_prev_Click(df, predictors, agg_suffix='prevClick', agg_type='float32'):

    agg_suffix='prevClick'

    agg_type='float32'

    print("Extracting {agg_suffix} time calculation features...\n")
    
    GROUP_BY_NEXT_CLICKS = [
    {'groupby': ['ip', 'channel']},
    
    #{'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    #{'groupby': ['ip', 'os', 'device']},
    #{'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        # print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df.click_time - df[all_features].groupby(spec[
                'groupby']).click_time.shift(+1) ).dt.seconds.astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
    return df, predictors 


def do_next_Click( df, predictors, agg_suffix='nextClick', agg_type='float32'):
    
    # print(f">> \nExtracting {agg_suffix} time calculation features...\n")
    
    GROUP_BY_NEXT_CLICKS = [
    
    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    # {'groupby': ['ip', 'channel']},
    # {'groupby': ['ip', 'os']},
    
    # V3
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    {'groupby': ['ip', 'os', 'device']},
    {'groupby': ['ip', 'os', 'device', 'app']},
    {'groupby': ['device']},
    {'groupby': ['device', 'channel']},     
    {'groupby': ['app', 'device', 'channel']},
    {'groupby': ['device', 'hour']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        # print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df[all_features].groupby(spec[
            'groupby']).click_time.shift(-1) - df.click_time).dt.seconds.astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
    return df, predictors 

def do_generating_nextClick(train_df, frm, to, debug, predictors):
    print('doing nextClick')
    
    new_feature = 'nextClick'
    filename='nextClick_%d_%d.csv'%(frm,to)

    if os.path.exists(filename):
        print('loading from save file')
        next_clicks=pd.read_csv(filename).values
    else:
        D=2**26
        train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
            + "_" + train_df['os'].astype(str)).apply(hash) % D
        click_buffer= np.full(D, 3000000000, dtype=np.uint32)

        train_df['epochtime']= train_df['click_time'].astype(np.int64) // 10 ** 9
        next_clicks= []
        for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
            next_clicks.append(click_buffer[category]-t)
            click_buffer[category]= t
        del(click_buffer)
        next_clicks= list(reversed(next_clicks))

        if not debug:
            print('saving')
            pd.DataFrame(next_clicks).to_csv(filename,index=False)
            
    train_df.drop(['epochtime','category','click_time'], axis=1, inplace=True)

    train_df[new_feature] = pd.Series(next_clicks).astype('float32')
    predictors.append(new_feature)

    train_df[new_feature+'_shift'] = train_df[new_feature].shift(+1).values
    predictors.append(new_feature+'_shift')
    
    del next_clicks
    gc.collect()
    return train_df, predictors

# do_preprocessing will process the pd.dataframe provide
def do_preprocessing(train_df, frm, to, debug, predictors):
    print('start pre-processing: ')
    train_df.info()
    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    train_df['wday'] = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8') # Which day of the week
    train_df = do_countuniq( train_df, ['ip'], 'channel', 'ip_channel_countuniq', 'uint8'); gc.collect()
    train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app', 'ip_dev_os_app_cumcount'); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'day'], 'hour', 'ip_day_hour_countuniq', 'uint8'); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'app', 'ip_app_countuniq', 'uint8'); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'app'], 'os', 'ip_app_os_countuniq', 'uint8'); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'device', 'ip_dev_countniq', 'uint16'); gc.collect()
    train_df = do_countuniq( train_df, ['app'], 'channel', 'app_channel_countuniq'); gc.collect()
    train_df = do_cumcount( train_df, ['ip'], 'os', 'ip_os_count'); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app', 'ip_dev_os_countuniq'); gc.collect()
    train_df = do_countuniq( train_df, ['app', 'os'], 'channel', 'app_os_channel_countuniq'); gc.collect()
    train_df = do_count( train_df, ['ip', 'day', 'hour'], 'ip_tcount'); gc.collect()
    train_df = do_count( train_df, ['ip', 'app'], 'ip_app_count'); gc.collect()
    train_df = do_count( train_df, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16'); gc.collect()
    train_df = do_count( train_df, ['ip', 'wday', 'hour'], 'ip_wday_hour_count'); gc.collect()
    train_df = do_count( train_df, ['ip', 'device', 'wday', 'hour'], 'ip_dev_wday_hour_count'); gc.collect()
    train_df = do_count( train_df, ['app', 'hour', 'wday'], 'app_hour_wday_count'); gc.collect()
    # train_df = do_var( train_df, ['ip', 'day', 'channel'], 'hour', 'ip_tchan_count'); gc.collect()
    # train_df = do_var( train_df, ['ip', 'app', 'channel'], 'day', 'ip_app_channel_var_day'); gc.collect()
    # train_df = do_var( train_df, ['ip', 'app'], 'channel', 'ip_app_channel_var'); gc.collect()
    

    train_df = do_var( train_df, ['ip', 'app', 'os'], 'hour', 'ip_app_os_var'); gc.collect()
    train_df = do_var( train_df, ['app', 'os'], 'channel', 'app_os_channel_var'); gc.collect()
    train_df = do_var( train_df, ['app', 'os'], 'wday', 'app_os_wday_var'); gc.collect()
    train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'hour', 'ip_app_channel_mean_hour'); gc.collect()
    train_df, predictors = do_prev_Click(train_df, predictors)
    train_df, predictors = do_next_Click(train_df, predictors)
    train_df, predictors = do_generating_nextClick(train_df, frm, to, debug, predictors)
    print('done pre-processing: ')
    # change data types of column
    train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
    train_df.info()
    gc.collect()
    return train_df, predictors