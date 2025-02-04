# IMPORTS
import numpy as np
import pandas as pd
import sklearn.metrics as mtr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, merge, Add,BatchNormalization
from keras.layers.embeddings import Embedding
from sklearn.model_selection import KFold,GroupKFold
from sklearn.preprocessing import LabelEncoder
import warnings
import random as rn
import tensorflow as tf
from keras.models import load_model
import keras
import os
import gc
from tqdm import tqdm
from scipy.spatial import Voronoi
import lightgbm as lgb
warnings.filterwarnings("ignore")


# from kaggle.competitions import nflrush
# env = nflrush.make_env()
# iter_test = env.iter_test()

train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})


outcomes = train[['GameId','PlayId','Yards']].drop_duplicates()


# evaluation metric
def crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])


# author : nlgn
# Link : https://www.kaggle.com/kingychiu/keras-nn-starter-crps-early-stopping
class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_train, y_train = self.data[0][0], self.data[0][1]
        y_pred = self.model.predict(X_train)
        y_true = np.clip(np.cumsum(y_train, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        tr_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_train[-1].shape[0])
        tr_s = np.round(tr_s, 6)
        logs['tr_CRPS'] = tr_s

        X_valid, y_valid = self.data[1][0], self.data[1][1]

        y_pred = self.model.predict(X_valid)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid[-1].shape[0])
        val_s = np.round(val_s, 6)
        logs['val_CRPS'] = val_s
        print('tr CRPS', tr_s, 'val CRPS', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)



# author : ryancaldwell
# Link : https://www.kaggle.com/ryancaldwell/location-eda
def create_features(df, deploy=False):
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
                                         .agg({'dist_to_back':['min','max','mean','std']})\
                                         .reset_index()
        player_distance.columns = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field',
                                   'min_dist','max_dist','mean_dist','std_dist']

        return player_distance


#     def defense_features(df):
#         rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
#         rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

#         df = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
#         defense = df[df['Team'] != df['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY','S','A','Dis','Orientation','Dir']]
#         defense['def_dist_to_back'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

#         defense = defense.groupby(['GameId','PlayId'])\
#                          .agg({'def_dist_to_back':['min','max','mean','std']})\
#                          .reset_index()
#         defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']

#         return defense


    def add_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

        df = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        df['dist_from_rusher'] = df[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
        df = df.sort_values(by=['GameId','PlayId','dist_from_rusher'])

        defense_summary = df[df['Team'] != df['RusherTeam']].groupby(['GameId','PlayId'])\
                         .agg({'dist_from_rusher':['min','max','mean','std']})\
                         .reset_index()
        defense_summary.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']

#         defense_closest = df[df['Team'] != df['RusherTeam']][::11][['GameId','PlayId','X','Y','S','A','Dis','Orientation','Dir']]  # Rusherに最も近いDefense
#         defense_closest.columns = ['GameId','PlayId','X_def_closest','Y_def_closest','S_def_closest','A_def_closest','Dis_def_closest','Orientation_def_closest','Dir_def_closest']

#         offense_closest = df[df['Team'] == df['RusherTeam']][1::11][['GameId','PlayId','X','Y','S','A','Dis','Orientation','Dir']]  # Rusherに最も近いOffense
#         offense_closest.columns = ['GameId','PlayId','X_off_closest','Y_off_closest','S_off_closest','A_off_closest','Dis_off_closest','Orientation_off_closest','Dir_def_closest']

#         df_nn = pd.merge(defense_summary, defense_closest, on=['GameId','PlayId'], how='inner')
#         df_nn = pd.merge(df_nn, offense_closest, on=['GameId','PlayId'], how='inner')

#         # Rusherのボロノイ領域を計算して特徴量を作成
#         # RusherがPlayIdの中で最も上に並ぶようにソート
#         # ソートしてから、ボロノイ領域を計算
#         vor = df.groupby(['GameId','PlayId'])[['X','Y']].apply(Voronoi)

#         # Rusherのボロノイ領域の面積
#         def PolyArea(arr):
#             x = arr[:,0]
#             y = arr[:,1]
#             return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

#         df_nn['X_maxInVoronoi'] = 0
#         df_nn['X_minInVoronoi'] = 0
#         df_nn['AreaInVoronoi'] = 0
#         for i, playid in enumerate(list(df_nn['PlayId'].unique())):
#             df_nn.loc[df_nn['PlayId']==playid, 'X_maxInVoronoi'] = vor.iloc[i].vertices[vor.iloc[i].regions[vor.iloc[i].point_region[0]]][:,0].max()  # Rusherのボロノイ領域の最大のx座標
#             df_nn.loc[df_nn['PlayId']==playid, 'X_minInVoronoi'] = vor.iloc[i].vertices[vor.iloc[i].regions[vor.iloc[i].point_region[0]]][:,0].min()  # Rusherのボロノイ領域の最小のx座標
#             df_nn.loc[df_nn['PlayId']==playid, 'AreaInVoronoi'] = PolyArea(vor.iloc[i].vertices[vor.iloc[i].regions[vor.iloc[i].point_region[0]]])  # Rusherのボロノイ領域の面積

        return defense_summary



    def static_features(df):
        static_features = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y','S','A','Dis','Orientation','Dir','OffenseFormation',
                                                            'YardLine','Quarter','Down','Distance','DefendersInTheBox','PlayerHeight','PlayerWeight','NflId','NflIdRusher','HomeScoreBeforePlay','VisitorScoreBeforePlay',
                                                            'DefensePersonnel',
                                                            'OffensePersonnel',
                                                            'Season']].drop_duplicates()
        static_features['DefendersInTheBox'] = static_features['DefendersInTheBox'].fillna(np.mean(static_features['DefendersInTheBox']))

        static_features['4th_Down'] = (static_features.Down==4) * 1
        static_features['Diff_Score'] = 0
        static_features.loc[(static_features['Team']=='home'), 'Diff_Score'] = static_features.loc[(static_features['Team']=='home'), 'HomeScoreBeforePlay'] - static_features.loc[(static_features['Team']=='home'), 'VisitorScoreBeforePlay']
        static_features.loc[(static_features['Team']=='away'), 'Diff_Score'] = static_features.loc[(static_features['Team']=='away'), 'VisitorScoreBeforePlay'] - static_features.loc[(static_features['Team']=='away'), 'HomeScoreBeforePlay']

        static_features['4th_above'] = ((static_features['Diff_Score'] > 0) & (static_features['4th_Down']==1)) * 1
        static_features['4th_behind'] = ((static_features['Diff_Score'] < 0) & (static_features['4th_Down']==1)) * 1

        static_features.drop(['Team','4th_Down','HomeScoreBeforePlay','VisitorScoreBeforePlay'], axis=1, inplace=True)

        return static_features

    def split_personnel(s):
        splits = s.split(',')
        for i in range(len(splits)):
            splits[i] = splits[i].strip()

        return splits

    def defense_formation(l):
        dl = 0
        lb = 0
        db = 0
        other = 0

        for position in l:
            sub_string = position.split(' ')
            if sub_string[1] == 'DL':
                dl += int(sub_string[0])
            elif sub_string[1] in ['LB','OL']:
                lb += int(sub_string[0])
            else:
                db += int(sub_string[0])

        counts = (dl,lb,db,other)

        return counts

    def offense_formation(l):
        qb = 0
        rb = 0
        wr = 0
        te = 0
        ol = 0

        sub_total = 0
        qb_listed = False
        for position in l:
            sub_string = position.split(' ')
            pos = sub_string[1]
            cnt = int(sub_string[0])

            if pos == 'QB':
                qb += cnt
                sub_total += cnt
                qb_listed = True
            # Assuming LB is a line backer lined up as full back
            elif pos in ['RB','LB']:
                rb += cnt
                sub_total += cnt
            # Assuming DB is a defensive back and lined up as WR
            elif pos in ['WR','DB']:
                wr += cnt
                sub_total += cnt
            elif pos == 'TE':
                te += cnt
                sub_total += cnt
            # Assuming DL is a defensive lineman lined up as an additional line man
            else:
                ol += cnt
                sub_total += cnt

        # If not all 11 players were noted at given positions we need to make some assumptions
        # I will assume if a QB is not listed then there was 1 QB on the play
        # If a QB is listed then I'm going to assume the rest of the positions are at OL
        # This might be flawed but it looks like RB, TE and WR are always listed in the personnel
        if sub_total < 11:
            diff = 11 - sub_total
            if not qb_listed:
                qb += 1
                diff -= 1
            ol += diff

        counts = (qb,rb,wr,te,ol)

        return counts

    def personnel_features(df):
        personnel = df[['GameId','PlayId','OffensePersonnel','DefensePersonnel']].drop_duplicates()
        personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: split_personnel(x))
        personnel['DefensePersonnel'] = personnel['DefensePersonnel'].apply(lambda x: defense_formation(x))
        personnel['num_DL'] = personnel['DefensePersonnel'].apply(lambda x: x[0])
        personnel['num_LB'] = personnel['DefensePersonnel'].apply(lambda x: x[1])
        personnel['num_DB'] = personnel['DefensePersonnel'].apply(lambda x: x[2])

        personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: split_personnel(x))
        personnel['OffensePersonnel'] = personnel['OffensePersonnel'].apply(lambda x: offense_formation(x))
        personnel['num_QB'] = personnel['OffensePersonnel'].apply(lambda x: x[0])
        personnel['num_RB'] = personnel['OffensePersonnel'].apply(lambda x: x[1])
        personnel['num_WR'] = personnel['OffensePersonnel'].apply(lambda x: x[2])
        personnel['num_TE'] = personnel['OffensePersonnel'].apply(lambda x: x[3])
        personnel['num_OL'] = personnel['OffensePersonnel'].apply(lambda x: x[4])

        # Let's create some features to specify if the OL is covered
        personnel['OL_diff'] = personnel['num_OL'] - personnel['num_DL']
        personnel['OL_TE_diff'] = (personnel['num_OL'] + personnel['num_TE']) - personnel['num_DL']
        # Let's create a feature to specify if the defense is preventing the run
        # Let's just assume 7 or more DL and LB is run prevention
        personnel['run_def'] = (personnel['num_DL'] + personnel['num_LB'] > 6).astype(int)

        personnel.drop(['OffensePersonnel','DefensePersonnel'], axis=1, inplace=True)

        return personnel

    def combine_features(relative_to_back, static, personnel, add_features, deploy=deploy):
        df = pd.merge(relative_to_back,static,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,personnel,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,add_features,on=['GameId','PlayId'],how='inner')

        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

        return df


    yardline = update_yardline(df)
    df = update_orientation(df, yardline)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    static_feats = static_features(df)
    personnel = personnel_features(df)

    add_features = add_features(df)

    basetable = combine_features(rel_back, static_feats, personnel, add_features, deploy=deploy)

#     display('rel_back', rel_back.head())
#     display('static_feats', static_feats.head())
#     display('personnel', personnel.head())
#     display('add_features', add_features.head())

    del rel_back, static_feats, personnel, add_features
    gc.collect()
    return basetable




train_basetable = create_features(train, False)

X = train_basetable.copy()
y = X.Yards + 99
X = X.drop(['Yards'], axis=1)

# y = np.zeros((yards.shape[0], 199))
# for idx, target in enumerate(list(yards)):
#     y[idx][99 + target] = 1


def process_two(t_):
    t_['fe1'] = pd.Series(np.sqrt(np.absolute(np.square(t_.X.values) + np.square(t_.Y.values))))
    t_['fe5'] = np.square(t_['S'].values) + 2 * t_['A'].values * t_['Dis'].values  # N
    t_['fe7'] = np.arccos(np.clip(t_['X'].values / t_['Y'].values, -1, 1))  # N
    t_['fe8'] = t_['S'].values / np.clip(t_['fe1'].values, 0.6, None)
    radian_angle = (90 - t_['Dir']) * np.pi / 180.0
    t_['fe10'] = np.abs(t_['S'] * np.cos(radian_angle))
    t_['fe11'] = np.abs(t_['S'] * np.sin(radian_angle))
    return t_

X = process_two(X)



cat = ['back_oriented_down_field', 'back_moving_down_field', 'OffenseFormation','4th_above','4th_behind'] # ,'DefensePersonnel','OffensePersonnel']

# num = ['back_from_scrimmage', 'min_dist', 'max_dist', 'mean_dist', 'std_dist',
#        'def_min_dist', 'def_max_dist', 'def_mean_dist', 'def_std_dist', 'X',
#        'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'YardLine']

# num_1 = ['back_from_scrimmage', 'min_dist', 'max_dist', 'mean_dist', 'std_dist',
#        'def_min_dist', 'def_max_dist', 'def_mean_dist', 'def_std_dist','YardLine']

# num_2 = ['X','Y', 'S', 'A', 'Dis', 'Orientation', 'Dir']

cols_rm = ['GameId','PlayId','Team','Season','PlayerHeight','PlayerWeight','NflId','NflIdRusher','HomeScoreBeforePlay','VisitorScoreBeforePlay','DefensePersonnel','OffensePersonnel']
num = [col for col in X.columns if col not in cat+cols_rm]

features = cat+num

le_list = []
for c in cat:
    X[c].fillna('NaN', inplace=True)
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c])
    X[c] = X[c].astype('category')
    le_list.append((c,le))




from sklearn.model_selection import GroupShuffleSplit
n_splits = 5
#kf = GroupKFold(n_splits=n_splits)
kf = GroupShuffleSplit(n_splits=n_splits, random_state=529)
score = []

#############################################################################

params = {
    'objective':'multiclass',
    'num_class': 199,
    "boosting": "gbdt",
    #"metric": 'mae',
    'learning_rate': 0.01,
    'num_leaves': 63,
    'max_depth': -1,
    # 'min_data_in_leaf': 30,
    # "min_child_samples": 20,
    # "feature_fraction": 0.9,
    # "bagging_freq": 1,
    # "bagging_fraction": 0.9 ,
    # "bagging_seed": 11,
    # "lambda_l1": 0.1,
    "verbosity": -1,
    "nthread": -1,
    "random_state": 16
}

models = []
oof = np.zeros((509762//22, 199))
feature_importance_df = pd.DataFrame()
for fold_, (tr_idx, val_idx) in enumerate(kf.split(X, y, X['GameId'])):
    print('-')
    print("Fold {}".format(fold_ + 1))

    tr_data = lgb.Dataset(X[features].iloc[tr_idx], label=y[tr_idx])
    val_data = lgb.Dataset(X[features].iloc[val_idx], label=y[val_idx])

    num_round = 10000
    model = lgb.train(params, tr_data, num_round, valid_sets = [tr_data, val_data], verbose_eval=100, early_stopping_rounds=100)
    oof[val_idx, :] = model.predict(X[features].iloc[val_idx], num_iteration=model.best_iteration)

    models.append(model)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = model.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


#############################################################################

y_ans = np.zeros((509762//22, 199))
for i,p in enumerate(y-99):
    p+=99
    for j in range(199):
        if j>=p:
            y_ans[i][j]=1.0

print("CV score:",np.sum(np.power(np.cumsum(oof, axis=1)-y_ans, 2))/(199*(509762//22)))

# output importance
feature_importance_mean = feature_importance_df[['feature','importance']].groupby('feature').mean().sort_values(by="importance", ascending=False)
print(feature_importance_mean.head(20))
feature_importance_mean.to_csv("../importance/lgbm.csv", index=True)

output = pd.read_csv('output.csv')
output_tmp = pd.DataFrame({'file': [os.path.basename(__file__)], 'CV_score': [np.mean(score)], 'LB': [np.nan]})
output = output.append(output_tmp)
output.to_csv('output.csv', index=False)
