#load gamelog_data
%matplotlib inline
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import time
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
from pyquery import PyQuery as pq
from bs4 import BeautifulSoup
import json
import requests
import datetime

def season_subset(df, year_season_start, year_season_end = None):
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    if year_season_end is None:
        year_season_end = year_season_start + 1
    df_gt = df[df.GAME_DATE > datetime.date(year_season_start,9,1)]
    df_lt = df_gt[df_gt.GAME_DATE < datetime.date(year_season_end,9,1)]
    return df_lt.sort_values("GAME_DATE") if not df_lt.empty else None

post85df = pd.read_csv('./gamelogs/master_post86df.csv')
post85df = post85df.drop('VIDEO_AVAILABLE',1)
df85_15 = season_subset(post85df,1985,2015)

by_player = df85_15.groupby("PLAYER_NAME")

df85_15["FANTASY_ZSCORE"] = by_player["FANTASY_PTS"].apply(lambda x: ((x - x.mean())/x.std()))
df85_15["i_ZSCORE_OVER"] = df85_15["FANTASY_ZSCORE"].map(lambda x: 1 if x > 1 else 0)
df85_15["SEASON_MIN"] = by_player['MIN'].apply(lambda x: x.map(lambda y: x.sum()))
df85_15["GAMES_PLAYED"] = by_player["PLAYER_NAME"].apply(lambda x: x.map(lambda y: len(x)))
for x in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
    df85_15[x] = df85_15[x].map(lambda y: 0 if np.isnan(y) else y)
df85_15["WL"] = [1 if v == "W" else 0 for v in df85_15.copy()["WL"]]

opp_home = df85_15.MATCHUP.map(lambda x: (x[-3:],0) if "@" in x else (x[-3:],1))
df85_15["OPP"] = opp_home.map(lambda x: x[0])
df85_15["i_HOME"] = opp_home.map(lambda x: x[1])

#Add player bio data for age,weight,height
player_bios_df = pd.read_csv("./player_bios/player_bios.csv")
player_bios_df = player_bios_df.rename(columns = {'PERSON_ID': 'PLAYER_ID', 'DISPLAY_FIRST_LAST': 'PLAYER_NAME'})
player_bios_df["BIRTHDATE"] = pd.to_datetime(player_bios_df["BIRTHDATE"])
player_bios_df['AGE'] = player_bios_df["BIRTHDATE"].map(lambda x: round((pd.to_datetime('today') - x).days / 365.,2))
player_bios_df["WEIGHT"] = player_bios_df["WEIGHT"].astype('str')
player_bios_df["HEIGHT"] = player_bios_df["HEIGHT"].astype('str')
player_bios_df["WEIGHT"] = player_bios_df["WEIGHT"].map(lambda x:  float(x) if x != 'nan' else 0.)
player_bios_df["HEIGHT"] = player_bios_df["HEIGHT"].map(lambda x: (12.*float(x[0]) + float(x[2:])) if x != 'nan' else 0.)

by_player = df85_15.groupby("PLAYER_NAME")

def get_player_bio(name, col_name):
    return float(player_bios_df[player_bios_df.PLAYER_NAME == name][col_name])

df85_15["AGE"] = by_player["PLAYER_NAME"].apply(lambda x: x.replace(x.iloc[0],get_player_bio(x.iloc[0],"AGE")))
df85_15["WEIGHT"] = by_player["PLAYER_NAME"].apply(lambda x: x.replace(x.iloc[0],get_player_bio(x.iloc[0],"WEIGHT")))
df85_15["HEIGHT"] = by_player["PLAYER_NAME"].apply(lambda x: x.replace(x.iloc[0],get_player_bio(x.iloc[0],"HEIGHT")))

#Integrate ELO Rankings
elo_df = pd.read_csv("./gamelogs/all_elo.csv")
elo_df["date_game"] = pd.to_datetime(elo_df["date_game"])
elo_df["game_location"] = elo_df["game_location"].map(lambda x: 1 if x == "H" else 0)
elo_df = elo_df[elo_df["is_playoffs"] == 0]

curr = elo_df.columns.tolist()
cols = [curr[i] for i in [5,8,11,13,14,17,19,21]]
elo_df = elo_df[cols]
elo_df = elo_df.rename(columns={'date_game': 'GAME_DATE',
                                'team_id':'TEAM_ABBREVIATION',
                                'opp_id':'OPP', 
                                'game_location': 'i_HOME',
                                'elo_i':'ELO',
                                'opp_elo_i': 'OPP_ELO',
                                'win_equiv': 'EXP_WINS',
                                'forecast':'FORECAST'})

df85_15 = df85_15.merge(season_subset(elo_df,1985,2015))

#Rearrange some columns in df85_15
curr = df85_15.columns.tolist()
cols = curr[:3] + curr[32:37] + curr[3:9] + curr[37:] + curr[9:32]
if len(curr) == len(cols):
    df85_15 = df85_15[cols]


name_pos = player_bios_df[["PLAYER_ID","POSITION","PLAYER_NAME"]]
df85_15 = df85_15.merge(name_pos)
df85_15.columns.tolist()

%connect_info
def calc_season_avg(df,col_list,(date_str1,date_str2)):
    date1, date2 = pd.to_datetime(date_str1), pd.to_datetime(date_str2)
    mask = lambda x: (date1 <= x) & (x <= date2)
    return df[df.GAME_DATE.apply(mask)].groupby(["PLAYER_NAME","SEASON_ID"])[col_list].mean().reset_index()

def ngames_colname(col_list, ngames):
    return map(lambda x: str(ngames) + 'D_' + x, col_list)


def last_ngames(df,ngames,game_date,col_list):
    ngames_df = df[df.GAME_DATE < game_date].nlargest(ngames, "GAME_DATE")
    ngames_col_list = ngames_colname(col_list,ngames)
    num_cols = len(ngames_col_list)
    date_player_tuples = [("GAME_DATE",game_date)]#,("PLAYER_NAME",df.PLAYER_NAME.iloc[0])]
    if ngames_df.empty:
        return dict(date_player_tuples + zip(ngames_col_list,np.array(0).repeat(num_cols)))
    else:
        return dict(date_player_tuples + zip(ngames_col_list,ngames_df[col_list].mean()))


def calc_ngame_avg(df,col_list,game_date_str,ngames):
    game_date = pd.to_datetime(game_date_str)
    season_id = df[df.GAME_DATE == game_date]["SEASON_ID"].iloc[0]
    return last_ngames(df[df.SEASON_ID == season_id],ngames,game_date,col_list)

def rolling_cols(df,col_list,ngames,rolling_kind):
    if rolling_kind == 'mean':
        rolling_func = lambda (a,b,c): pd.rolling_mean(a,b,min_periods = c)
    elif rolling_kind == 'sum':
        rolling_func = lambda (a,b,c): pd.rolling_sum(a,b,min_periods = c)
    else:
        return None 
    
    rolling_df = (df.groupby(["PLAYER_NAME","SEASON_ID"])
                    .apply(lambda x: add_game_date_pts_col(rolling_func((x[col_list],ngames,1)),x.GAME_DATE,x.FANTASY_PTS).reset_index(drop = True)))
    return rolling_df.reset_index().drop('level_2',axis = 1).rename(columns=dict(zip(col_list,map(lambda x: 'R_' + x,col_list))))

def add_game_date_pts_col(df,game_date_col,fantasy_pts_col):
   df["GAME_DATE"] = game_date_col
   df["FANTASY_PTS"] = fantasy_pts_col
   return df

def per_season_cumsum(df,col_list):
    cumsum_df = (df.groupby(["PLAYER_NAME","SEASON_ID"])
                   .apply(lambda x: add_game_date_col(x[col_list].cumsum(axis = 0), x.GAME_DATE).reset_index(drop = True)))
    return cumsum_df.reset_index().drop('level_2',axis = 1).rename(columns=dict(zip(col_list,map(lambda x: 'C_' + x,col_list))))

def per_season_cummean(df,col_list):
    cumsum_df = (df.groupby(["PLAYER_NAME","SEASON_ID"])
                   .apply(lambda x: add_game_date_pts_col(pd.expanding_mean(x[col_list], min_periods = 2), x.GAME_DATE, x.FANTASY_PTS).reset_index(drop = True)))
    return cumsum_df.reset_index().drop('level_2',axis = 1).rename(columns=dict(zip(col_list,map(lambda x: 'C_' + x,col_list))))


def get_player_seasons(player_name, season1,season2,full_df):
    player_df = (full_df[full_df.PLAYER_NAME == player_name].groupby(["PLAYER_NAME","SEASON_ID"])
                                .apply(lambda x: pd.DataFrame(map(lambda y: calc_ngame_avg(x.sort_values("GAME_DATE"),["AST","REB","PTS","TOV","STL","BLK"],y,3),x.GAME_DATE)))).reset_index().drop('level_2',axis = 1)
    player_df = pd.merge(player_df,full_df[full_df.PLAYER_NAME == player_name][["GAME_DATE","FANTASY_PTS","OPP_ELO"]])
    player_df['SHIT'] = player_df['OPP_ELO'].map(lambda x: 1 if x < 1400 else 0)
    player_df['OKAY'] = player_df['OPP_ELO'].map(lambda x: 1 if 1400 <= x < 1600 else 0)
    player_df['GOOD'] = player_df['OPP_ELO'].map(lambda x: 1 if 1600 <= x < 1700 else 0)
    player_df['GREAT'] = player_df['OPP_ELO'].map(lambda x: 1 if 1700 <= x else 0)
    player_df2 = player_df.set_index('GAME_DATE')
    fantasy_resp = player_df2.groupby('SEASON_ID').apply(lambda x: x['FANTASY_PTS'].map(lambda y: 1 if y > x.FANTASY_PTS.mean() else 0)).reset_index().rename(columns={'FANTASY_PTS':'FANTASY_RESP'})
    player_df2 = pd.merge(player_df2,fantasy_resp)
    fst_season = season1 + 20000
    lst_season = season2 + 20000
    player_df_final = player_df2[(player_df2.SEASON_ID <= lst_season) & (player_df2.SEASON_ID >= fst_season)].sort_values('SEASON_ID')
    return (player_df_final, np.array(player_df_final.SEASON_ID < lst_season))

df,mask = get_player_seasons("Roy Hibbert",2010,2013,df85_15) 
df.head()
from sklearn.cross_validation import train_test_split
#train_test_split(xrange(df.shape[0]), train_size=0.7)
mask.shape,mask.sum()

dftouse = df.copy()

STANDARDIZABLE = map(lambda x: '3D_' + x,["AST","REB","PTS","TOV","STL","BLK"])
from sklearn.preprocessing import StandardScaler
for col in STANDARDIZABLE:
    print col
    valstrain=df[col].values[mask]
    valstest=df[col].values[~mask]
    scaler=StandardScaler().fit(valstrain)
    outtrain=scaler.transform(valstrain)
    outtest=scaler.fit_transform(valstest)
    out=np.empty(mask.shape[0])
    out[mask]=outtrain
    out[~mask]=outtest
    dftouse[col]=out

lcols = STANDARDIZABLE + ["SHIT","OKAY","GOOD","GREAT"]

from sklearn import svm 
clfsvm = svm.SVC(kernel = 'rbf')
cs=[.01,.1,1,10,100]
gammas = range(0,101,5)
Xmatrix=dftouse[lcols].values
Yresp=dftouse['FANTASY_RESP'].values 
Xmatrix_train=Xmatrix[mask]
Xmatrix_test=Xmatrix[~mask]
Yresp_train=Yresp[mask]
Yresp_test=Yresp[~mask]
df[~mask].tail()

#your code here
from sklearn.grid_search import GridSearchCV
gs=GridSearchCV(clfsvm, param_grid={'C':cs,'gamma':gammas}, cv=5)
gs.fit(Xmatrix_train, Yresp_train)
print "BEST", gs.best_params_, gs.best_score_, gs.grid_scores_

#calculate the accuracy here, bullshit
best = gs.best_estimator_
best.fit(Xmatrix_train, Yresp_train)
best.score(Xmatrix_test, Yresp_test)

