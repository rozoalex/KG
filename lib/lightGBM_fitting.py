import lightgbm as lgb
import gc
import time


# For Chen Li
def lgb_modelfit_nocv(dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=30, num_boost_round=1000, verbose_eval=1, categorical_features=None):

    lgb_params = {
        'boosting_type': 'gbdt', # gbdt, goss, 
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.1, # 0.05
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 7,  # 31 # 200
        'max_depth': 3,  # -1 means no limit # 8 
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf) # 100
        'max_bin': 100,  # Number of bucketed bin for feature values # 100
        'subsample': 0.7,  # Subsample ratio of the training instance. # 0.8
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable # 0
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree. #0.9
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf) # 5
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4, # Changed from 4 to 8 
        'verbose': 0, # 0
        'scale_pos_weight': 200 # 200
    }

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    print("lgb_params:")
    print(lgb_params)

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])

    return (bst1,bst1.best_iteration)
# For Chen Li 