
import math
import numpy as np
import pandas as pd
import sklearn.metrics as mtr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,GroupKFold,StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization,Concatenate
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from  keras.callbacks import EarlyStopping,ModelCheckpoint
import datetime
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")


TRAIN_OFFLINE = True

# train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
if TRAIN_OFFLINE:
    train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
else:
    train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})

outcomes = train[['GameId','PlayId','Yards']].drop_duplicates()


def strtofloat(x):
    try:
        return float(x)
    except:
        return -1

def orientation_to_cat(x):
    x = np.clip(x, 0, 360 - 1)
    try:
        return str(int(x/15))
    except:
        return "nan"

def new_X(x_coordinate, play_direction):
        if play_direction == 'left':
            return 120.0 - x_coordinate
        else:
            return x_coordinate

def new_line(rush_team, field_position, yardline):
    if rush_team == field_position:
        # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
        return 10.0 + yardline
    else:
        # half the field plus the yards between midfield and the line of scrimmage
        return 60.0 + (50 - yardline)

def new_orientation(angle, play_direction):
    if play_direction == 'left':
        new_angle = 360.0 - angle
        if new_angle == 360.0:
            new_angle = 0.0
        return new_angle
    else:
        return angle

def euclidean_distance(x1,y1,x2,y2):
    x_diff = (x1-x2)**2
    y_diff = (y1-y2)**2

    return np.sqrt(x_diff + y_diff)

def back_direction(orientation):
    if orientation > 180.0:
        return 1
    else:
        return 0

def update_yardline(df):
    new_yardline = df[df['NflId'] == df['NflIdRusher']]
    new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)
    new_yardline = new_yardline[['GameId','PlayId','YardLine']]

    return new_yardline

def update_orientation(df, yardline):
    df['X'] = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)
    df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
    df['Dir'] = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)

    df = df.drop('YardLine', axis=1)
    df = pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')

    return df

def back_features(df):
    carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher','X','Y','Orientation','Dir','YardLine']]
    carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
    carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: back_direction(x))
    carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: back_direction(x))
    carriers = carriers.rename(columns={'X':'back_X',
                                        'Y':'back_Y'})
    carriers = carriers[['GameId','PlayId','NflIdRusher','back_X','back_Y','back_from_scrimmage','back_oriented_down_field','back_moving_down_field']]

    return carriers

def features_relative_to_back(df, carriers):
    player_distance = df[['GameId','PlayId','NflId','X','Y']]
    player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
    player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
    player_distance['dist_to_back'] = player_distance[['X','Y','back_X','back_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

    player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field'])\
                                        .agg({'dist_to_back':['min','max','mean','std'],
                                        'X':['std'],
                                        'Y':['std']})\
                                        .reset_index()
    player_distance.columns = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field','min_dist','max_dist','mean_dist','std_dist','X_std','Y_std']

    return player_distance

def defense_features(df):
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
    rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

    df = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
    defense = df[df['Team'] != df['RusherTeam']][['GameId','PlayId','X','Y','S','Dir','RusherX','RusherY']]
    defense['dist_from_rusher'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

    df_summary = defense.groupby(['GameId','PlayId'])\
                        .agg({'dist_from_rusher':['min','max','mean','median','std'],
                            'X':['mean','median','std'],
                            'Y':['mean','median','std'],
                            'S':['median'],
                            'Dir':['median']})\
                        .reset_index()
    df_summary.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_med_dist','def_std_dist','def_X_mean','def_X_med','def_X_std','def_Y_mean','def_Y_med','def_Y_std','def_S_med','def_Dir_med']


    #####
    offense = df[df['Team'] == df['RusherTeam']][['GameId','PlayId','NflId','NflIdRusher','X','Y','S','Dir','RusherX','RusherY']]
    offense['dist_from_rusher'] = offense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
    offense_summary = offense.groupby(['GameId','PlayId'])\
                        .agg({
                            'dist_from_rusher': ['max','mean','median','std'],
                            'X': ['mean','median','std'],
                            'Y': ['mean','median','std'],
                            'S': ['median'],
                            'Dir':['median']
                        })\
                        .reset_index()
    offense_summary.columns = ['GameId','PlayId','off_max_dist','off_mean_dist','off_med_dist','off_std_dist','off_X_mean','off_X_med','off_X_std','off_Y_mean','off_Y_med','off_Y_std','off_S_med','off_Dir_med']

    offense_close = offense[(offense['NflId']!=offense['NflIdRusher'])].groupby(['GameId','PlayId'])\
                    .agg({
                        'dist_from_rusher': ['min'],
                    })\
                    .reset_index()
    offense_close.columns = ['GameId','PlayId','off_min_dist']


    df_summary = pd.merge(df_summary, offense_summary, on=['GameId','PlayId'])
    df_summary = pd.merge(df_summary, offense_close, on=['GameId','PlayId'])


    df_summary['dist_between_def_off'] = df_summary[['def_X_mean','def_Y_mean','off_X_mean','off_Y_mean']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
    df_summary['dist_between_def_off'] = df_summary['dist_between_def_off'] / np.sqrt(df_summary['def_std_dist'] * df_summary['off_std_dist'])


    df_summary['def_X'] = df_summary['def_X_med'] + df_summary['def_S_med'] * df_summary['def_Dir_med']
    df_summary['def_Y'] = df_summary['def_Y_med'] + df_summary['def_S_med'] * df_summary['def_Dir_med']
    df_summary['off_X'] = df_summary['off_X_med'] + df_summary['off_S_med'] * df_summary['off_Dir_med']
    df_summary['off_Y'] = df_summary['off_Y_med'] + df_summary['off_S_med'] * df_summary['off_Dir_med']


    # df_summary['diff_dist_def_max_med'] = df_summary['def_max_dist'] - df_summary['def_med_dist']
    # df_summary['diff_dist_off_max_med'] = df_summary['off_max_dist'] - df_summary['off_med_dist']
    # df_summary['diff_dist_def_max_med_rate'] = df_summary['diff_dist_off_max_med'] / df_summary['diff_dist_def_max_med']
    # df_summary['diff_Y_def_mean_med'] = df_summary['def_Y_mean'] - df_summary['def_Y_med']
    # df_summary['diff_Y_off_mean_med'] = df_summary['off_Y_mean'] - df_summary['off_Y_med']
    # df_summary['diff_Y_def_max_med_rate'] = df_summary['diff_Y_def_mean_med'] / df_summary['diff_Y_off_mean_med']

    df_summary.drop(['def_med_dist','off_med_dist','def_X_mean','def_X_med','def_Y_mean','def_Y_med','off_X_mean','off_X_med', 'off_Y_mean','off_Y_med'], axis=1, inplace=True)

    return df_summary

def rusher_features(df):

    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Dir', 'S', 'A', 'X', 'Y']]
    rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS', 'RusherA', 'RusherX', 'RusherY']


    radian_angle = (90 - rusher['RusherDir']) * np.pi / 180.0
    v_horizontal = np.abs(rusher['RusherS'] * np.cos(radian_angle))
    v_vertical = np.abs(rusher['RusherS'] * np.sin(radian_angle))


    rusher['v_horizontal'] = v_horizontal
    rusher['v_vertical'] = v_vertical


    rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS','RusherA','RusherX', 'RusherY','v_horizontal', 'v_vertical']


    return rusher


def create_features(df, deploy=False):

    def static_features(df):


        add_new_feas = []

        ## Height
        df['PlayerHeight_dense'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

        add_new_feas.append('PlayerHeight_dense')

        ## Time
        df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
        df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

        df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
        df['PlayerBirthDate'] =df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

        ## Age
        seconds_in_year = 60*60*24*365.25
        df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
        add_new_feas.append('PlayerAge')

        ## WindSpeed
        df['WindSpeed_ob'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
        df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
        df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
        df['WindSpeed_dense'] = df['WindSpeed_ob'].apply(strtofloat)
        add_new_feas.append('WindSpeed_dense')

        ## Orientation and Dir
        df["Orientation_ob"] = df["Orientation"].apply(lambda x : orientation_to_cat(x)).astype("object")
        df["Dir_ob"] = df["Dir"].apply(lambda x : orientation_to_cat(x)).astype("object")

        df["Orientation_sin"] = df["Orientation"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
        df["Orientation_cos"] = df["Orientation"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
        df["Dir_sin"] = df["Dir"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
        df["Dir_cos"] = df["Dir"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
        add_new_feas.append("Dir_sin")
        add_new_feas.append("Dir_cos")

        ## diff Score
        df["diffScoreBeforePlay"] = df["HomeScoreBeforePlay"] - df["VisitorScoreBeforePlay"]
        add_new_feas.append("diffScoreBeforePlay")

        # # Formation
        # df['OffenseFormation_SINGLEBACK'] = (df['OffenseFormation']=='SINGLEBACK') * 1
        # df['OffenseFormation_SHOTGUN'] = (df['OffenseFormation']=='SHOTGUN') * 1
        # df['OffenseFormation_I_FORM'] = (df['OffenseFormation']=='I_FORM') * 1
        # df['OffenseFormation_JUMBO'] = (df['OffenseFormation']=='JUMBO') * 1
        # add_new_feas.extend(["OffenseFormation_SINGLEBACK","OffenseFormation_SHOTGUN","OffenseFormation_I_FORM","OffenseFormation_JUMBO"])


        static_features = df[df['NflId'] == df['NflIdRusher']][add_new_feas+[
            'GameId','PlayId','X','Y','S','A','Dis','Orientation','Dir','YardLine','Quarter','Down','Distance','DefendersInTheBox']].drop_duplicates()

        static_features.fillna(-999,inplace=True)

        return static_features


    def combine_features(relative_to_back, defense,rushing, static, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,rushing,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')

        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

        return df

    yardline = update_yardline(df)
    df = update_orientation(df, yardline)

    ##########################################################
    ##########################################################
    # Dir -> radianに
    df['Dir'] = np.mod(90 - df.Dir, 360) * math.pi/180.0
    df['X'] += df['S'] * np.cos(df['Dir'])
    df['Y'] += df['S'] * np.sin(df['Dir'])
    ##########################################################
    ##########################################################

    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    rush_feats = rusher_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats,rush_feats,static_feats, deploy=deploy)

     # # 特徴量作成
    basetable['def_mean_rate'] = basetable['def_mean_dist'] / basetable['mean_dist']  # 全体の平均の中での～
    basetable['off_mean_rate'] = basetable['off_mean_dist'] / basetable['mean_dist']



    return basetable


def create_features_2(df, deploy=False):

    def static_features(df):
        static_features = df[df['NflId'] == df['NflIdRusher']][[
            'GameId','PlayId','X','Y']].drop_duplicates()

        static_features.fillna(-999,inplace=True)

        return static_features


    def combine_features(relative_to_back, defense,rushing, static, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,rushing,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')

        return df

    yardline = update_yardline(df)
    df = update_orientation(df, yardline)

    # ##########################################################
    # ##########################################################
    # # Dir -> radianに
    # df['Dir'] = np.mod(90 - df.Dir, 360) * math.pi/180.0
    # df['X'] += df['S'] * np.cos(df['Dir'])
    # df['Y'] += df['S'] * np.sin(df['Dir'])
    # ##########################################################
    # ##########################################################

    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    rush_feats = rusher_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats,rush_feats,static_feats, deploy=deploy)

     # # 特徴量作成
    basetable['def_mean_rate'] = basetable['def_mean_dist'] / basetable['mean_dist']  # 全体の平均の中での～
    basetable['off_mean_rate'] = basetable['off_mean_dist'] / basetable['mean_dist']


    basetable.columns = [col+'_2' if col not in ['GameId','PlayId'] else col for col in basetable.columns]

    return basetable


def create_features_3(df, deploy=False):

    def static_features(df):
        static_features = df[df['NflId'] == df['NflIdRusher']][[
            'GameId','PlayId','X','Y']].drop_duplicates()

        static_features.fillna(-999,inplace=True)

        return static_features


    def combine_features(relative_to_back, defense,rushing, static, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,rushing,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')

        return df

    yardline = update_yardline(df)
    df = update_orientation(df, yardline)

    ##########################################################
    ##########################################################
    # Dir -> radianに
    df['Dir'] = np.mod(90 - df.Dir, 360) * math.pi/180.0
    df['X'] += 1.5*df['S'] * np.cos(df['Dir'])
    df['Y'] += 1.5*df['S'] * np.sin(df['Dir'])
    #df['S'] += 0.5*df['A']
    ##########################################################
    ##########################################################

    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    rush_feats = rusher_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats,rush_feats,static_feats, deploy=deploy)

     # # 特徴量作成
    basetable['def_mean_rate'] = basetable['def_mean_dist'] / basetable['mean_dist']  # 全体の平均の中での～
    basetable['off_mean_rate'] = basetable['off_mean_dist'] / basetable['mean_dist']



    basetable.columns = [col+'_3' if col not in ['GameId','PlayId'] else col for col in basetable.columns]
    return basetable


def create_features_4(df, deploy=False):

    def static_features(df):
        static_features = df[df['NflId'] == df['NflIdRusher']][[
            'GameId','PlayId','X','Y']].drop_duplicates()

        static_features.fillna(-999,inplace=True)

        return static_features


    def combine_features(relative_to_back, defense,rushing, static, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,rushing,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')

        return df

    yardline = update_yardline(df)
    df = update_orientation(df, yardline)

    ##########################################################
    ##########################################################
    # Dir -> radianに
    df['Dir'] = np.mod(90 - df.Dir, 360) * math.pi/180.0
    df['X'] += 0.5*df['S'] * np.cos(df['Dir'])
    df['Y'] += 0.5*df['S'] * np.sin(df['Dir'])
    #df['S'] += 0.5*df['A']
    ##########################################################
    ##########################################################

    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    rush_feats = rusher_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats,rush_feats,static_feats, deploy=deploy)

    # # 特徴量作成
    basetable['def_mean_rate'] = basetable['def_mean_dist'] / basetable['mean_dist']  # 全体の平均の中での～
    basetable['off_mean_rate'] = basetable['off_mean_dist'] / basetable['mean_dist']


    basetable.columns = [col+'_4' if col not in ['GameId','PlayId'] else col for col in basetable.columns]

    return basetable


def create_features_5(df, deploy=False):

    def static_features(df):
        static_features = df[df['NflId'] == df['NflIdRusher']][[
            'GameId','PlayId','X','Y']].drop_duplicates()

        static_features.fillna(-999,inplace=True)

        return static_features


    def combine_features(relative_to_back, defense,rushing, static, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,rushing,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')

        return df

    yardline = update_yardline(df)
    df = update_orientation(df, yardline)

    ##########################################################
    ##########################################################
    # Dir -> radianに
    df['Dir'] = np.mod(90 - df.Dir, 360) * math.pi/180.0
    df['X'] += 10*df['S'] * np.cos(df['Dir'])
    df['Y'] += 10*df['S'] * np.sin(df['Dir'])
    #df['S'] += 0.5*df['A']
    ##########################################################
    ##########################################################

    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    rush_feats = rusher_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats,rush_feats,static_feats, deploy=deploy)

    # # 特徴量作成
    basetable['def_mean_rate'] = basetable['def_mean_dist'] / basetable['mean_dist']  # 全体の平均の中での～
    basetable['off_mean_rate'] = basetable['off_mean_dist'] / basetable['mean_dist']


    basetable.columns = [col+'_5' if col not in ['GameId','PlayId'] else col for col in basetable.columns]

    return basetable



train_basetable = create_features(train, False)
train_basetable_2 = create_features_2(train, False)
train_basetable_3 = create_features_3(train, False)
train_basetable_4 = create_features_4(train, False)
train_basetable_5 = create_features_5(train, False)
train_basetable = pd.merge(train_basetable, train_basetable_2, on=['GameId','PlayId'])
train_basetable = pd.merge(train_basetable, train_basetable_3, on=['GameId','PlayId'])
train_basetable = pd.merge(train_basetable, train_basetable_4, on=['GameId','PlayId'])
train_basetable = pd.merge(train_basetable, train_basetable_5, on=['GameId','PlayId'])


X = train_basetable.copy()
yards = X.Yards


y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1

X.drop(['GameId','PlayId','Yards'], axis=1, inplace=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)



# #######################################################################
# # LGBM
# import lightgbm as lgb

# param = {'num_leaves': 63,
#          'min_data_in_leaf': 30,
#          'objective':'multiclass',
#          'num_class': 199,
#          'max_depth': 6, # -1
#          'learning_rate': 0.01,
#          "min_child_samples": 20,
#          "boosting": "gbdt",
#          "feature_fraction": 0.4, #0.7
#          "bagging_freq": 1,
#          "bagging_fraction": 0.9,
#          "bagging_seed": 11,
#          "metric": 'multi_logloss',
#          "lambda_l1": 0.1,
#          "verbosity": -1,
#          'n_jobs': -1,
#          "seed":1234}


# models_lgb = []
# oof = np.zeros((509762//22, 199))
# y_lgb = yards+99
# if  TRAIN_OFFLINE==False:
#     for k in range(2):
#         kfold = KFold(5, random_state = 42 + k, shuffle = True)
#         for k_fold, (tr_idx, val_idx) in enumerate(kfold.split(X, y_lgb)):
#             print(f'Fold : {k_fold+1}')
#             tr_data = lgb.Dataset(X[tr_idx], label=y_lgb[tr_idx])
#             val_data = lgb.Dataset(X[val_idx], label=y_lgb[val_idx])

#             num_round = 10000
#             model_lgb = lgb.train(param, tr_data, num_round, valid_sets = [tr_data, val_data], verbose_eval=100, early_stopping_rounds=100)
#             oof[val_idx,:] = model_lgb.predict(X[val_idx], num_iteration=model_lgb.best_iteration)
#             models_lgb.append(model_lgb)
# else:
#     kfold = KFold(5, random_state = 42, shuffle = True)
#     for k_fold, (tr_idx, val_idx) in enumerate(kfold.split(X, y_lgb)):
#         print(f'Fold : {k_fold+1}')
#         tr_data = lgb.Dataset(X[tr_idx], label=y_lgb[tr_idx])
#         val_data = lgb.Dataset(X[val_idx], label=y_lgb[val_idx])

#         num_round = 10000
#         model_lgb = lgb.train(param, tr_data, num_round, valid_sets = [tr_data, val_data], verbose_eval=100, early_stopping_rounds=100)
#         oof[val_idx,:] = model_lgb.predict(X[val_idx], num_iteration=model_lgb.best_iteration)
#         models_lgb.append(model_lgb)


# y_ans = np.zeros((509762//22, 199))
# for i,p in enumerate(y_lgb-99):
#     p+=99
#     for j in range(199):
#         if j>=p:
#             y_ans[i][j]=1.0

# y_oof = np.cumsum(oof, axis=1).clip(0,1)
# #y_oof[:,:85] = 0
# print("LGBM CVscore:",np.sum(np.power(y_oof-y_ans, 2))/(199*(509762//22)))

# #######################################################################


from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda
from keras.models import Model
import keras.backend as K
import re
from keras.losses import binary_crossentropy
from  keras.callbacks import EarlyStopping,ModelCheckpoint
import codecs

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score




class CRPSCallback(Callback):

    def __init__(self,validation, predict_batch_size=20, include_on_batch=False):
        super(CRPSCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

        #print('validation shape',len(self.validation))

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('CRPS_score_val' in self.params['metrics']):
            self.params['metrics'].append('CRPS_score_val')

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['CRPS_score_val'] = float('-inf')

    def on_epoch_end(self, epoch, logs={}):
        logs['CRPS_score_val'] = float('-inf')

        if (self.validation):
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_pred = self.model.predict(X_valid)
            y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
            y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
            val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
            val_s = np.round(val_s, 6)
            logs['CRPS_score_val'] = val_s



def get_model(x_tr,y_tr,x_val,y_val):
    inp = Input(shape = (x_tr.shape[1],))
    x = Dense(1024, input_dim=X.shape[1], activation='relu')(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    out = Dense(199, activation='softmax')(x)
    model = Model(inp,out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
    #add lookahead
#     lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
#     lookahead.inject(model) # add into model

    if  TRAIN_OFFLINE==False:
        es = EarlyStopping(monitor='CRPS_score_val',
                        mode='min',
                        restore_best_weights=True,
                        verbose=2,
                        patience=10)
    else:
        es = EarlyStopping(monitor='CRPS_score_val',
                        mode='min',
                        restore_best_weights=True,
                        verbose=2,
                        patience=5)

    mc = ModelCheckpoint('best_model.h5',monitor='CRPS_score_val',mode='min',
                                   save_best_only=True, verbose=2, save_weights_only=True)

    bsz = 1024
    steps = x_tr.shape[0]/bsz



    model.fit(x_tr, y_tr,callbacks=[CRPSCallback(validation = (x_val,y_val)),es,mc], epochs=100, batch_size=bsz,verbose=False)
    model.load_weights("best_model.h5")

    y_pred = model.predict(x_val)
    y_valid = y_val
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * x_val.shape[0])
    crps = np.round(val_s, 6)

    return model,crps


from sklearn.model_selection import train_test_split, KFold
import time

losses = []
models = []
crps_csv = []

if  TRAIN_OFFLINE==False:
    for k in range(2):
        kfold = KFold(5, random_state = 42 + k, shuffle = True)
        for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(X, yards)):
            print("-----------")
            print("-----------")
            tr_x,tr_y = X[tr_inds],y[tr_inds]
            val_x,val_y = X[val_inds],y[val_inds]

            model,crps = get_model(tr_x,tr_y,val_x,val_y)
            models.append(model)
            print("the %d fold crps is %f"%((k_fold+1),crps))
            crps_csv.append(crps)


else:
    kfold = KFold(5, random_state = 42, shuffle = True)
    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(X, yards)):
        print("-----------")
        print("-----------")
        tr_x,tr_y = X[tr_inds],y[tr_inds]
        val_x,val_y = X[val_inds],y[val_inds]

        model,crps = get_model(tr_x,tr_y,val_x,val_y)
        models.append(model)
        print("the %d fold crps is %f"%((k_fold+1),crps))
        crps_csv.append(crps)


print("mean crps is %f"%np.mean(crps_csv))



def predict(x_te):
    model_num = len(models)
    for k,m in enumerate(models):
        if k==0:
            y_pred = m.predict(x_te,batch_size=1024)
        else:
            y_pred+=m.predict(x_te,batch_size=1024)
    y_pred = y_pred / model_num

    # LGBM
    for l, model in enumerate(models_lgb):
        if l==0:
            y_pred_lgb = model.predict(x_te, num_iteration=model.best_iteration) / len(models_lgb)
        else:
            y_pred_lgb += model.predict(x_te, num_iteration=model.best_iteration) / len(models_lgb)

    y_pred = y_pred*0.7 + y_pred_lgb * 0.3

    return y_pred



if  TRAIN_OFFLINE==False:
    from kaggle.competitions import nflrush
    env = nflrush.make_env()
    iter_test = env.iter_test()

    for (test_df, sample_prediction_df) in tqdm(iter_test):
        basetable = create_features(test_df, deploy=True)
        basetable_2 = create_features_2(test_df, deploy=True)
        basetable_3 = create_features_3(test_df, deploy=True)
        basetable_4 = create_features_4(test_df, deploy=True)

        basetable = pd.merge(basetable, basetable_2, on=['GameId','PlayId'])
        basetable = pd.merge(basetable, basetable_3, on=['GameId','PlayId'])
        basetable = pd.merge(basetable, basetable_4, on=['GameId','PlayId'])

        basetable.drop(['GameId','PlayId'], axis=1, inplace=True)
        scaled_basetable = scaler.transform(basetable)

        y_pred = predict(scaled_basetable)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]

        preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
        env.predict(preds_df)

    env.write_submission_file()

else:
    output = pd.read_csv('output1124_.csv')
    output_tmp = pd.DataFrame({'file': [os.path.basename(__file__)], 'CV_score': [np.mean(crps_csv)], 'LB': [np.nan]})
    output = output.append(output_tmp)
    output.to_csv('output1124_.csv', index=False)
