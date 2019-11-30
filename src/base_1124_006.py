
import math
import numpy as np
import pandas as pd
import sklearn.metrics as mtr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization
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

    def defense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

        df = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        defense = df[df['Team'] != df['RusherTeam']][['GameId','PlayId','X','Y','S','Dir','RusherX','RusherY']]
        defense['dist_from_rusher'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        df_summary = defense.groupby(['GameId','PlayId'])\
                         .agg({'dist_from_rusher':['min','max','mean','median','std'],
                              'X':['mean','median'],
                              'Y':['mean','median'],
                              'S':['mean'],
                              'Dir':['mean']})\
                         .reset_index()
        df_summary.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_med_dist','def_std_dist','def_X_mean','def_X_med', 'def_Y_mean','def_Y_med','def_S_mean','def_Dir_mean']


        #####
        offense = df[df['Team'] == df['RusherTeam']][['GameId','PlayId','NflId','NflIdRusher','X','Y','S','Dir','RusherX','RusherY']]
        offense['dist_from_rusher'] = offense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
        offense_summary = offense.groupby(['GameId','PlayId'])\
                            .agg({
                                'dist_from_rusher': ['max','mean','median','std'],
                                'X': ['mean','median'],
                                'Y': ['mean','median'],
                                'S': ['mean'],
                                'Dir':['mean']
                            })\
                            .reset_index()
        offense_summary.columns = ['GameId','PlayId','off_max_dist','off_mean_dist','off_med_dist','off_std_dist','off_X_mean','off_X_med', 'off_Y_mean','off_Y_med','off_S_mean','off_Dir_mean']

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


        df_summary['def_X'] = df_summary['def_X_mean'] + df_summary['def_S_mean'] * df_summary['def_Dir_mean']
        df_summary['def_Y'] = df_summary['def_Y_mean'] + df_summary['def_S_mean'] * df_summary['def_Dir_mean']
        df_summary['off_X'] = df_summary['off_X_mean'] + df_summary['off_S_mean'] * df_summary['off_Dir_mean']
        df_summary['off_Y'] = df_summary['off_Y_mean'] + df_summary['off_S_mean'] * df_summary['off_Dir_mean']


        # df_summary['diff_dist_def_max_med'] = df_summary['def_max_dist'] - df_summary['def_med_dist']
        # df_summary['diff_dist_off_max_med'] = df_summary['off_max_dist'] - df_summary['off_med_dist']
        # df_summary['diff_dist_def_max_med_rate'] = df_summary['diff_dist_off_max_med'] / df_summary['diff_dist_def_max_med']
        # df_summary['diff_Y_def_mean_med'] = df_summary['def_Y_mean'] - df_summary['def_Y_med']
        # df_summary['diff_Y_off_mean_med'] = df_summary['off_Y_mean'] - df_summary['off_Y_med']
        # df_summary['diff_Y_def_max_med_rate'] = df_summary['diff_Y_def_mean_med'] / df_summary['diff_Y_off_mean_med']

        df_summary.drop(['def_med_dist','off_med_dist','def_X_mean','def_X_med','def_Y_mean','def_Y_med','off_X_mean','off_X_med', 'off_Y_mean','off_Y_med','def_S_mean','off_S_mean','def_Dir_mean','off_Dir_mean'], axis=1, inplace=True)

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

    # # 特徴量作成 # #
    basetable['def_mean_rate'] = basetable['def_mean_dist'] / basetable['mean_dist']  # 全体の平均の中での～
    basetable['off_mean_rate'] = basetable['off_mean_dist'] / basetable['mean_dist']

    basetable['X_def_rate'] = basetable['X'] / basetable['def_X']
    basetable['Y_def_rate'] = basetable['Y'] / basetable['def_Y']
    basetable['X_def_off_rate'] = basetable['off_X'] / basetable['def_X']
    basetable['Y_def_off_rate'] = basetable['off_Y'] / basetable['def_Y']

    return basetable


train_basetable = create_features(train, False)

X = train_basetable.copy()
yards = X.Yards

y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1

X.drop(['GameId','PlayId','Yards'], axis=1, inplace=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)


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

s_time = time.time()

if  TRAIN_OFFLINE==False:
    for k in range(2):
        kfold = KFold(5, random_state = 42 + k, shuffle = True)
        for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
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
    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
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

    return y_pred



if  TRAIN_OFFLINE==False:
    from kaggle.competitions import nflrush
    env = nflrush.make_env()
    iter_test = env.iter_test()

    for (test_df, sample_prediction_df) in tqdm(iter_test):
        basetable = create_features(test_df, deploy=True)
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

