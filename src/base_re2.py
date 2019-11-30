# IMPORTS
import math
import numpy as np
import pandas as pd
import sklearn.metrics as mtr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, merge, Add
from keras.layers.embeddings import Embedding
from sklearn.model_selection import KFold,GroupKFold
from sklearn.preprocessing import LabelEncoder
import warnings
import random as rn
import tensorflow as tf
from keras.models import load_model
import os
import gc
from tqdm import tqdm
import re
from string import punctuation
from scipy.spatial import Voronoi
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



def std_table(df, deploy=False):

    def new_X(x_coordinate, play_direction):
        """
        全てのプレーを、左から右に攻撃が進むように、x座標を正規化
        """
        if play_direction == 'left':
            return 120.0 - x_coordinate
        else:
            return x_coordinate

    def new_line(rush_team, field_position, yardline):
        """
        YardLineを、座標に直す
        """
        if rush_team == field_position:
            # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
            return 10.0 + yardline
        else:
            # half the field plus the yards between midfield and the line of scrimmage
            return 60.0 + (50 - yardline)

    def new_orientation(angle, play_direction):
        """
        左から右へ攻撃が進むように、角度も正規化
        """
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


        # 2017年と、2018年以降で、Orientationの定義が変化
        # https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python
        df['Orientation_rad'] = np.mod(df.Orientation, 360) * math.pi/180.0
        df.loc[df.Season >= 2018, 'Orientation_rad'] = np.mod(df.loc[df.Season >= 2018, 'Orientation'] - 90, 360) * math.pi/180.0

        df = df.drop(['Orientation','YardLine'], axis=1)
        df = df.rename(columns={'Orientation_rad': 'Orientation'})
        df = pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')

        return df

    yardline = update_yardline(df)
    df = update_orientation(df, yardline)

    return df

def correction_df(df):

    def correction_wind(df):
        ws_to_wd = {"SSW": 13, "E": 8, "SE": 1}
        wd_to_ws = {13: "SSW", 8: "E", 1: "SE"}


        def clean_ws(x):
            x = str(x).upper().replace(" ", "").replace("MPH", "")
            if "GUSTSUPTO" in x:
                return np.mean(np.array(x.split("GUSTSUPTO"), dtype=int))
            elif "-" in x:
                return np.mean(np.array(x.split("-"), dtype=int))
            elif x == "CALM" or x == "99":
                return 0.0
            else:
                try:
                    return float(x)
                except ValueError:
                    return x


        def clean_wd(x):
            try:
                return float(x)
            except ValueError:
                return x


        df["WindSpeed"] = df["WindSpeed"].fillna(99).apply(lambda x: clean_ws(x))
        df["WindDirection"] = df["WindDirection"].apply(lambda x: clean_wd(x))
        df["WindSpeed"] = [
            ws_to_wd[ws] if ws in ws_to_wd.keys() else ws for ws in df["WindSpeed"]
        ]
        df["WindDirection"] = [
            wd_to_ws[wd] if wd in wd_to_ws.keys() else wd for wd in df["WindDirection"]
        ]

#         print(f"WindSpeed: {df['WindSpeed'].unique()}")
#         print(f"WindDirection: {df['WindDirection'].unique()}")

        return df


    map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
    for abb in df['PossessionTeam'].unique():
        map_abbr[abb] = abb

    def clean_StadiumType(txt):
        if pd.isna(txt):
            return np.nan
        txt = txt.lower()
        txt = ''.join([c for c in txt if c not in punctuation])
        txt = re.sub(' +', ' ', txt)
        txt = txt.strip()
        txt = txt.replace('outside', 'outdoor')
        txt = txt.replace('outdor', 'outdoor')
        txt = txt.replace('outddors', 'outdoor')
        txt = txt.replace('outdoors', 'outdoor')
        txt = txt.replace('oudoor', 'outdoor')
        txt = txt.replace('indoors', 'indoor')
        txt = txt.replace('ourdoor', 'outdoor')
        txt = txt.replace('retractable', 'rtr.')
        return txt

    def transform_StadiumType(txt):
        if pd.isna(txt):
            return np.nan
        if 'outdoor' in txt or 'open' in txt:
            return str(1)
        if 'indoor' in txt or 'closed' in txt:
            return str(0)

        return np.nan

    #from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087
    Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial',
            'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural',
            'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural',
            'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial',
            'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'}

    def height2cm(h):
        return int(h[0]) * 30.48 + int(h[2]) * 2.58

    def str_to_float(txt):
        try:
            return float(txt)
        except:
            return -1

    def clean_WindDirection(txt):
        if pd.isna(txt):
            return np.nan
        txt = txt.lower()
        txt = ''.join([c for c in txt if c not in punctuation])
        txt = txt.replace('from', '')
        txt = txt.replace(' ', '')
        txt = txt.replace('north', 'n')
        txt = txt.replace('south', 's')
        txt = txt.replace('west', 'w')
        txt = txt.replace('east', 'e')
        return txt

    def transform_WindDirection(txt):
        if pd.isna(txt):
            return np.nan

        if txt=='n':
            return 0
        if txt=='nne' or txt=='nen':
            return 1/8
        if txt=='ne':
            return 2/8
        if txt=='ene' or txt=='nee':
            return 3/8
        if txt=='e':
            return 4/8
        if txt=='ese' or txt=='see':
            return 5/8
        if txt=='se':
            return 6/8
        if txt=='ses' or txt=='sse':
            return 7/8
        if txt=='s':
            return 8/8
        if txt=='ssw' or txt=='sws':
            return 9/8
        if txt=='sw':
            return 10/8
        if txt=='sww' or txt=='wsw':
            return 11/8
        if txt=='w':
            return 12/8
        if txt=='wnw' or txt=='nww':
            return 13/8
        if txt=='nw':
            return 14/8
        if txt=='nwn' or txt=='nnw':
            return 15/8
        return np.nan


    def map_weather(txt):
        """
        climate controlled or indoor => 3, sunny or sun => 2, clear => 1, cloudy => -1, rain => -2, snow => -3, others => 0
        partly => multiply by 0.5
        """
        ans = 1
        if pd.isna(txt):
            return 0
        if 'partly' in txt:
            ans*=0.5
        if 'climate controlled' in txt or 'indoor' in txt:
            return ans*3
        if 'sunny' in txt or 'sun' in txt:
            return ans*2
        if 'clear' in txt:
            return ans
        if 'cloudy' in txt:
            return -ans
        if 'rain' in txt or 'rainy' in txt:
            return -2*ans
        if 'snow' in txt:
            return -3*ans
        return 0


    #########################################
    # formation features
    #########################################
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

        df = pd.merge(df, personnel, on=['GameId','PlayId'])

        return df


    df = correction_wind(df)

    df['StadiumType'] = df['StadiumType'].apply(clean_StadiumType)
    df['StadiumType'] = df['StadiumType'].apply(transform_StadiumType)

    df['Turf'] = df['Turf'].map(Turf)
    df['Turf'] = (df['Turf'] == 'Natural') * 1

    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)

    df['HomePossesion'] = (df['PossessionTeam'] == df['HomeTeamAbbr']) * 1
    df['Field_eq_Possession'] = (df['FieldPosition'] == df['PossessionTeam']) * 1
    df['HomeField'] = (df['FieldPosition'] == df['HomeTeamAbbr']) * 1

    t_gemeclock = pd.to_datetime(df['GameClock'])
    df['GameClock'] = t_gemeclock.dt.minute*60 + t_gemeclock.dt.second

    df['PlayerHeight'] = df['PlayerHeight'].map(height2cm)

    df['PlayerBirthDate'] = pd.to_datetime(df['PlayerBirthDate'])
    df['Age'] = 2019 - df['PlayerBirthDate'].dt.year

    t_handoff = pd.to_datetime(df['TimeHandoff'])
    t_handsnap = pd.to_datetime(df['TimeSnap'])
    df['TimeDelta'] = (t_handoff.dt.minute*60 + t_handoff.dt.second) - (t_handsnap.dt.minute*60 + t_handsnap.dt.second)

    # remove mph
    # replace the ones that has x-y by (x+y)/2
    # and also the ones with x gusts up to y
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: str(x).lower().replace('mph', '').strip() if not pd.isna(x) else x)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
    df['WindSpeed'] = df['WindSpeed'].apply(str_to_float)

    df['WindDirection'] = df['WindDirection'].apply(clean_WindDirection)
    df['WindDirection'] = df['WindDirection'].apply(transform_WindDirection)

    #########################################
    # winddirection
    #########################################
    map_wind = {1/8:15/8, 2/8:14/8, 3/8:13/8, 4/8:12/8, 5/8:11/8, 6/8:10/8, 7/8:9/8}
    df.loc[df.PlayDirection=='left', 'WindDirection'] =  df.loc[df.PlayDirection=='left', 'WindDirection'].map(map_wind)

    # binary variables to 0/1
    #df['Team'] = (df['Team'].apply(lambda x: x.strip()=='home')) * 1

    df['GameWeather'] = df['GameWeather'].str.lower()
    indoor = "indoor"
    df['GameWeather'] = df['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
    df['GameWeather'] = df['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
    df['GameWeather'] = df['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
    df['GameWeather'] = df['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    df['GameWeather'] = df['GameWeather'].apply(map_weather)

    df['IsRusher'] = (df['NflId'] == df['NflIdRusher']) * 1

    df['YardsLeft'] = df.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)
    df['YardsLeft'] = df.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)

    df = personnel_features(df)

    return df


def back_direction(orientation):
        if orientation > 180.0:
            return 1
        else:
            return 0

def static_features(df):
    static_features = df[df['NflId'] == df['NflIdRusher']][[
        'GameId','PlayId','Season','Team','X','Y','S','A','Dis','Orientation','Dir','OffenseFormation',
        'YardLine','Quarter','Down','Distance','DefendersInTheBox','PlayerHeight','PlayerWeight','NflId','NflIdRusher','HomeScoreBeforePlay','VisitorScoreBeforePlay',
        'DefensePersonnel','OffensePersonnel',
        'Week','StadiumType','Turf','GameWeather','Temperature','Humidity','WindSpeed','WindDirection','Age','TimeDelta','YardsLeft',
        'num_DL','num_LB','num_DB','num_QB','num_RB','num_WR','num_TE','num_OL','OL_diff','OL_TE_diff','run_def'
        ]].drop_duplicates()

    static_features['DefendersInTheBox'] = static_features['DefendersInTheBox'].fillna(np.mean(static_features['DefendersInTheBox']))

    static_features['back_from_scrimmage'] = static_features['YardLine'] - static_features['X']
    static_features['back_oriented_down_field'] = static_features['Orientation'].apply(lambda x: back_direction(x))
    static_features['back_moving_down_field'] = static_features['Dir'].apply(lambda x: back_direction(x))

    static_features['Diff_Score'] = 0
    static_features.loc[(static_features['Team']=='home'), 'Diff_Score'] = static_features.loc[(static_features['Team']=='home'), 'HomeScoreBeforePlay'] - static_features.loc[(static_features['Team']=='home'), 'VisitorScoreBeforePlay']
    static_features.loc[(static_features['Team']=='away'), 'Diff_Score'] = static_features.loc[(static_features['Team']=='away'), 'VisitorScoreBeforePlay'] - static_features.loc[(static_features['Team']=='away'), 'HomeScoreBeforePlay']

    return static_features


def euclidean_distance(x1,y1,x2,y2):
    x_diff = (x1-x2)**2
    y_diff = (y1-y2)**2
    return np.sqrt(x_diff + y_diff)

def features_relative_to_rusher(df):
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
    rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

    df = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
    df['dist_from_rusher'] = df[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
    #df = df.sort_values(by=['GameId','PlayId','Team','dist_from_rusher'])

    defense_summary = df[df['Team'] != df['RusherTeam']].groupby(['GameId','PlayId'])\
                     .agg({'dist_from_rusher':['min','max','mean','std']})\
                     .reset_index()
    defense_summary.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']

    offense_summary = df[(df['Team'] == df['RusherTeam']) & (df['NflId'] != df['NflIdRusher'])].groupby(['GameId','PlayId'])\
                     .agg({'dist_from_rusher':['min','max','mean','std']})\
                     .reset_index()
    offense_summary.columns = ['GameId','PlayId','off_min_dist','off_max_dist','off_mean_dist','off_std_dist']

    df_summary = pd.merge(defense_summary, offense_summary, on=['GameId','PlayId'])
    df_summary['def/off_min_dist'] = df_summary['def_min_dist'] / df_summary['off_min_dist']
    df_summary['def/off_min_mean_dist'] = df_summary['def_min_dist'] / df_summary['off_mean_dist']
    df_summary['off/def_min_mean_dist'] = df_summary['off_min_dist'] / df_summary['def_mean_dist']
    df_summary['def/off_max_dist'] = df_summary['def_max_dist'] / df_summary['off_max_dist']
    df_summary['def/off_max_mean_dist'] = df_summary['def_max_dist'] / df_summary['off_mean_dist']
    df_summary['off/def_max_mean_dist'] = df_summary['off_max_dist'] / df_summary['def_mean_dist']
    df_summary['def/off_max_min_dist'] = df_summary['def_max_dist'] / df_summary['off_min_dist']
    df_summary['off/def_max_min_dist'] = df_summary['off_max_dist'] / df_summary['def_min_dist']


#     # Rusherのボロノイ領域を計算して特徴量を作成
#     # Rusherとdefenseの12人で計算
#     # RusherがPlayIdの中で最も上に並ぶようにソート,ソートしてから、ボロノイ領域を計算
#     df = df[(df['NflId'] == df['NflIdRusher']) | (df['Team'] != df['RusherTeam'])].sort_values(by=['GameId','PlayId','dist_from_rusher'])
#     vor = df.groupby(['GameId','PlayId'])[['X','Y']].apply(Voronoi)

#     # Rusherのボロノイ領域の面積
#     def PolygonSort(corners):
#         n = len(corners)
#         cx = float(sum(x for x, y in corners)) / n
#         cy = float(sum(y for x, y in corners)) / n
#         cornersWithAngles = []
#         for x, y in corners:
#             an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
#             cornersWithAngles.append((x, y, an))
#         cornersWithAngles.sort(key = lambda tup: tup[2])
#         cornersWithAngles = [(x,y) for (x,y,an) in cornersWithAngles]
#         return cornersWithAngles

#     def PolygonArea(corners):
#         n = len(corners)
#         area = 0.0
#         for i in range(n):
#             j = (i + 1) % n
#             area += corners[i][0] * corners[j][1]
#             area -= corners[j][0] * corners[i][1]
#         area = abs(area) / 2.0
#         return area

#     df_summary['AreaInVoronoi'] = 0
#     for i, playid in enumerate(list(df_summary['PlayId'].unique())):
# #         df_summary.loc[df_summary['PlayId']==playid, 'X_maxInVoronoi'] = vor.iloc[i].vertices[vor.iloc[i].regions[vor.iloc[i].point_region[0]]][:,0].max()  # Rusherのボロノイ領域の最大のx座標
# #         df_summary.loc[df_summary['PlayId']==playid, 'X_minInVoronoi'] = vor.iloc[i].vertices[vor.iloc[i].regions[vor.iloc[i].point_region[0]]][:,0].min()  # Rusherのボロノイ領域の最小のx座標
#         df_summary.loc[df_summary['PlayId']==playid, 'AreaInVoronoi'] = PolygonArea(PolygonSort(vor.iloc[i].vertices[vor.iloc[i].regions[vor.iloc[i].point_region[0]]]))  # Rusherのボロノイ領域の面積

    return df_summary



train = std_table(train, False)  # 向きの正規化
train = correction_df(train)  # textの正規化、数値化

static_features = static_features(train)
features_relative_to_rusher = features_relative_to_rusher(train)
train_basetable = pd.merge(static_features, features_relative_to_rusher, on=['GameId','PlayId'])

train_basetable = pd.merge(train_basetable, outcomes, on=['GameId','PlayId'], how='inner')

del train, static_features, features_relative_to_rusher
gc.collect()


X = train_basetable.copy()
yards = X.Yards
X = X.drop(['Yards'], axis=1)

y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1


# def process_two(t_):
#     t_['fe1'] = pd.Series(np.sqrt(np.absolute(np.square(t_.X.values) + np.square(t_.Y.values))))
#     t_['fe5'] = np.square(t_['S'].values) + 2 * t_['A'].values * t_['Dis'].values  # N
#     t_['fe7'] = np.arccos(np.clip(t_['X'].values / t_['Y'].values, -1, 1))  # N
#     t_['fe8'] = t_['S'].values / np.clip(t_['fe1'].values, 0.6, None)
#     radian_angle = (90 - t_['Dir']) * np.pi / 180.0
#     t_['fe10'] = np.abs(t_['S'] * np.cos(radian_angle))
#     t_['fe11'] = np.abs(t_['S'] * np.sin(radian_angle))
#     return t_

# X = process_two(X)


cat = ['back_oriented_down_field', 'back_moving_down_field','OffenseFormation','OffensePersonnel','DefensePersonnel','StadiumType','Turf']

num = ['back_from_scrimmage',
       'def_min_dist', 'def_max_dist', 'def_mean_dist', 'def_std_dist', 'X',
       'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'YardLine']

num = num + ['off_min_dist','off_max_dist','off_mean_dist','off_std_dist'] + ['def/off_min_dist','def/off_min_mean_dist','off/def_min_mean_dist']

num = num + ['Down','Distance','DefendersInTheBox','Week','Temperature','Humidity','WindSpeed','WindDirection','Age','TimeDelta','GameWeather','YardsLeft','num_DL','num_LB','num_DB','num_QB','num_RB','num_WR','num_TE','num_OL','OL_diff','OL_TE_diff','run_def']

# 'FieldPosition','TimeHandoff','TimeSnap','PlayerCollegeName','Position','HomePossesion','Field_eq_Possession','HomeField'


le_list = []
for c in cat:
    X[c].fillna('NaN', inplace=True)
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c])
    le_list.append((c,le))


scaler = StandardScaler()
X[num] = X[num].fillna(X[num].mean())
X[num] = scaler.fit_transform(X[num])


def model_NN():
    inputs = []
    embeddings = []
    for i in cat:
        input_ = Input(shape=(1,))
        embedding = Embedding(int(np.absolute(X[i]).max() + 1), 10, input_length=1)(input_)
        embedding = Reshape(target_shape=(10,))(embedding)
        inputs.append(input_)
        embeddings.append(embedding)
    input_numeric = Input(shape=(len(num),))
    embedding_numeric = Dense(2048, activation='relu')(input_numeric)
    inputs.append(input_numeric)
    embeddings.append(embedding_numeric)
    x = Concatenate()(embeddings)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(199, activation='softmax')(x)
    model = Model(inputs, output)
    return model


n_splits = 5
kf = GroupKFold(n_splits=n_splits)
score = []
for i_, (tdx, vdx) in enumerate(kf.split(X, y, X['GameId'])):
    print(f'Fold : {i_+1}')
    X_train, X_val, y_train, y_val = X.iloc[tdx], X.iloc[vdx], y[tdx], y[vdx]
    X_train = [np.absolute(X_train[i]) for i in cat] + [X_train[num]]
    X_val = [np.absolute(X_val[i]) for i in cat] + [X_val[num]]
    model = model_NN()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[])
    es = EarlyStopping(monitor='val_CRPS',
                   mode='min',
                   restore_best_weights=True,
                   verbose=2,
                   patience=5)
    es.set_model(model)
    metric = Metric(model, [es], [(X_train,y_train), (X_val,y_val)])
    for i in range(1):
        model.fit(X_train, y_train, verbose=False)
    for i in range(1):
        model.fit(X_train, y_train, batch_size=64, verbose=False)
    for i in range(1):
        model.fit(X_train, y_train, batch_size=128, verbose=False)
    for i in range(1):
        model.fit(X_train, y_train, batch_size=256, verbose=False)
    model.fit(X_train, y_train, callbacks=[metric], epochs=100, batch_size=1024, verbose=False)
    score_ = crps(y_val, model.predict(X_val))
    model.save(f'keras_{i_}.h5')
    print(score_)
    score.append(score_)


print('CV score: ', np.mean(score))

output = pd.DataFrame({'file': [os.path.basename(__file__)], 'CV_score': [np.mean(score)], 'LB': [0.1369]})
output.to_csv('output.csv', index=False)
