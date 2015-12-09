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

df85_15["BIRTHDATE"] = by_player["PLAYER_NAME"].apply(lambda x: x.replace(x.iloc[0],get_player_bio(x.iloc[0],"BIRTHDATE")))
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
    
    rolling_df = (df.groupby(["PLAYER_NAME","SEASON_ID","OPP"])
                    .apply(lambda x: add_game_date_pts_col(rolling_func((x[col_list],ngames,2)),x.GAME_DATE,x.OPP).reset_index(drop = True)))
    return rolling_df.reset_index().drop('level_2',axis = 1).rename(columns=dict(zip(col_list,map(lambda x: 'R_' + x,col_list))))

def add_game_date_pts_col(df,game_date_col,opp_col):
   new_df = pd.concat([df,game_date_col,opp_col], axis = 1)
   return new_df

def per_season_cumsum(df,col_list):
    cumsum_df = (df.groupby(["PLAYER_NAME","SEASON_ID"])
                   .apply(lambda x: add_game_date_col(x[col_list].cumsum(axis = 0), x.GAME_DATE).reset_index(drop = True)))
    return cumsum_df.reset_index().drop('level_2',axis = 1).rename(columns=dict(zip(col_list,map(lambda x: 'C_' + x,col_list))))

def per_season_cummean(df,col_list):
    cumsum_df = (df.groupby(["PLAYER_NAME","SEASON_ID"])
                   .apply(lambda x: add_game_date_pts_col(pd.expanding_mean(x[col_list], min_periods = 2), x.GAME_DATE, x.OPP).reset_index(drop = True)))
    return cumsum_df.reset_index().drop('level_2',axis = 1).rename(columns=dict(zip(col_list,map(lambda x: 'C_' + x,col_list))))

def enumerate_games(df):
    new_df = df.copy()
    new_df["GAME_NUM"] = range(1,len(df.GAME_DATE) + 1)
    return new_df

def sigmoidfun(x):
	return 1/(1+np.exp(-0.007*(x-800)))

def fantasy_avg_lastn(player_df,last_n_seasons,seasons):
    return player_df[[s in seasons[-last_n_seasons:] for s in player_df.SEASON_ID]]['FANTASY_PTS'].mean()    

def true_fantasy_mean(player_df,last_n_seasons):
    seasons = list(set(player_df.SEASON_ID))
    lastn_mean = fantasy_avg_lastn(player_df,last_n_seasons,seasons)
    return player_df.groupby("SEASON_ID").apply(lambda x: x.apply(lambda y: lastn_mean + sigmoidfun(y.MIN) * (y.C_FANTASY_PTS - lastn_mean),axis = 1))

def fantasy_resp(df):
    return df.groupby('PLAYER_NAME').apply(lambda x: true_fantasy_mean(x,5))

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

def calc_melo(x):
    MIN_PER = calc_melo_MIN_PER(x)
    TRUE_PER = calc_melo_TRUE_PER(x)
    FT_FRQ = calc_melo_FT_FRQ(x)
    MELO_HT = calc_melo_height(x)
    MELO_WT = calc_melo_weight(x)
    MELO_C_MIN = calc_melo_C_MIN(x)
    MELO_FT_PER = calc_melo_FT_PER(x)
    MELO_SCORE = MIN_PER + TRUE_PER + FT_FRQ + MELO_HT + MELO_WT +
    MELO_C_MIN + MELO_FT_PER
    return MELO_SCORE

def CM_HEIGHT(x):
    MELO_HT = x['HEIGHT'] * 3
    return 'MELO_HT',MELO_HT

def CM_WEIGHT(x):
    MELO_WT = x['WEIGHT'] * 3
    return 'MELO_WT',MELO_WT

def CM_MIN_PER(x):
    MIN_PER = (x['SEASON_MIN']/x['GAMES_PLAYED']) * 3.5
    return 'MELO_MIN_PER',MIN_PER

def CM_TRUE_PER(x):
    TRUE_PER = ((x['PTS'])/(x['FGA'] + x['FTM'] * 0.44)) * 5
    return 'MELO_TRUE_PER',TRUE_PER

def CM_FT_FRQ(x):
    FT_FRQ = (x['FTA']/x['FGA']) * 1.5
    return 'MELO_FT_FRQ',FT_FRQ

def CM_C_MIN(x):
    MELO_C_MIN = x['SEASON_MIN'] * 1.5
    return 'MELO_C_MIN',MELO_C_MIN

def CM_FT_PER(x):
    MELO_FT_PER = x['FT_PCT'] * 2.5
    return 'MELO_FT_PER',MELO_FT_PER

def fab_melo(player):
    root = df85_15[df85_15.PLAYER_NAME == player].sort_values('GAME_DATE')
    #new_root = root.groupby("SEASON_ID").apply(lambda x: calc_melo(x)).reset_index().rename(columns={0:'FAB_MELO_SCORE'})
    #new_root['MELO_HEIGHT'] = calc_melo_height(root)
    #new_root['MELO_WEIGHT'] = calc_melo_weight(root)
    #new_root['MELO_C_MIN'] = calc_melo_C_MIN(root)
    #new_root['PLAYER_NAME'] = [player] * new_root.shape[0]
    calc_melo_funcs = [CM_HEIGHT, CM_WEIGHT, CM_MIN_PER,CM_TRUE_PER,CM_FT_FRQ,CM_C_MIN,CM_FT_PER]
    return root.groupby('SEASON_ID').apply(lambda x: pd.DataFrame(dict([('SEASON_ID',x.SEASON_ID),('PLAYER_NAME',x.PLAYER_NAME),('GAME_DATE',x.GAME_DATE)] + map(lambda y: y(x),calc_melo_funcs))))

fab_melo("Kobe Bryant")
store_df = []
players = set(season_subset(df85_15,1996,2015)['PLAYER_NAME'])
for player in players:
    store_df.append(fab_melo(player))
FAB_MELO = pd.concat(store_df,axis = 0)
FAB_MELO


def get_player_seasons(player_name, season1,season2,full_df,ewma_colresp, ewma_colfeat):
    ewma_pos = full_df.groupby(["OPP",'SEASON_ID',"POSITION","GAME_DATE"]).apply(lambda x: x.FANTASY_PTS.sum()) 
    ewma_pos_df_temp = ewma_pos.reset_index().rename(columns={0:'TOT_OPP_POS'}).sort_values('GAME_DATE').groupby(["OPP",'SEASON_ID',"POSITION"]).apply(lambda x: pd.DataFrame(zip(x.GAME_DATE,np.log(pd.ewma(x.TOT_OPP_POS, span = 3).shift(1) + 1)), index = range(x.shape[0]))).rename(columns={0:'GAME_DATE',1:'EWMA_OPP_POS'}).reset_index(level = [0,1,2])
    ewma_pos_df = pd.merge(full_df,ewma_pos_df_temp,left_on=['OPP','GAME_DATE','POSITION','SEASON_ID'], right_on=['OPP','GAME_DATE','POSITION','SEASON_ID'])
    
    player_df = ewma_pos_df[ewma_pos_df.PLAYER_NAME == player_name].copy()
    player_df['SHIT'] = player_df['OPP_ELO'].map(lambda x: 1 if x < 1400 else 0)
    player_df['OKAY'] = player_df['OPP_ELO'].map(lambda x: 1 if 1400 <= x < 1600 else 0)
    player_df['GOOD'] = player_df['OPP_ELO'].map(lambda x: 1 if 1600 <= x < 1700 else 0)
    player_df['GREAT'] = player_df['OPP_ELO'].map(lambda x: 1 if 1700 <= x else 0)
    player_df2 = pd.concat([player_df.reset_index(drop = True),player_df.groupby("SEASON_ID").apply(lambda x: np.log(pd.ewma(x[ewma_colresp], span = 3).shift(1) + 1).reset_index().drop('index',axis=1).rename(columns={ewma_colresp:'EWMA_LOG_' + ewma_colresp})).reset_index(drop = True)],axis = 1)
    for ewma_col in ewma_colfeat:
        player_df2['EWMA_LOG_' + ewma_col] = player_df2.groupby("SEASON_ID").apply(lambda x: np.log(pd.ewma(x[ewma_col], span = 3).shift(1) + 1).reset_index().drop('index',axis=1).rename(columns={ewma_col:'EWMA_LOG_' + ewma_col})).reset_index(drop = True)
    resp = player_df2.groupby('SEASON_ID').apply(lambda x: x.apply(lambda y: 1 if abs(np.log(y[ewma_colresp] + 1) - y['EWMA_LOG_' + ewma_colresp]) < .2 else 0, axis = 1).reset_index().drop('index',axis=1).rename(columns={0: ewma_colresp + '_RESP'})).reset_index(drop = True)
    player_df3 = pd.concat([player_df2,resp], axis = 1)
    fst_season = season1 + 20000
    lst_season = season2 + 20000
    player_df_final = player_df3[(player_df3.SEASON_ID <= lst_season) & (player_df3.SEASON_ID >= fst_season)].dropna()
    return player_df_final, np.array(player_df_final.SEASON_ID < lst_season)

def classify_players_ondate(df,game_date, ewma_colresp, ewma_colfeats):
    players = list(set(df[df.GAME_DATE == game_date]['PLAYER_NAME']))
    sub_df = df[(df.GAME_DATE <= game_date) & (df.SEASON_ID >= season1)]
    ewma_pos = sub_df.groupby(["OPP",'SEASON_ID',"POSITION","GAME_DATE"]).apply(lambda x: x.FANTASY_PTS.sum()) 
    ewma_pos_df_temp = (ewma_pos.reset_index().rename(columns={0:'TOT_OPP_POS'})
                                .sort_values('GAME_DATE')
                                .groupby(["OPP",'SEASON_ID',"POSITION"])
                                .apply(lambda x: 
                                    pd.DataFrame(zip(x.GAME_DATE,np.log(pd.ewma(x.TOT_OPP_POS, span = 3).shift(1) + 1)), 
                                    index = range(x.shape[0])))
                                .rename(columns={0:'GAME_DATE',1:'EWMA_OPP_POS'})
                                .reset_index(level = [0,1,2]))
    ewma_pos_df = (pd.merge(sub_df,ewma_pos_df_temp,left_on=['OPP','GAME_DATE','POSITION','SEASON_ID'], 
                                                    right_on=['OPP','GAME_DATE','POSITION','SEASON_ID']))

    store_df = []
    for player in players[:2]:
        print player
    store_df.append(reduce_picks(player,game_date, ewma_pos_df, ewma_colresp, ewma_colfeats))
return pd.concat(store_df,axis = 0)


league_avg_df = ewma_pos_df.groupby(["SEASON_ID",'POSITION']).apply(lambda x: x['EWMA_OPP_POS'].mean()).reset_index().rename(columns={0:'LEAGUE_AVG_POS'})
nan_dict = dict(reduce(lambda x,y: x + y.items(),[{(k1,k2):v} for k1,k2,v in league_avg_df.to_records(index = False)], []))
ewma_pos_df.loc[pd.isnull(ewma_pos_df['EWMA_OPP_POS']),'EWMA_OPP_POS'] = ewma_pos_df[pd.isnull(ewma_pos_df['EWMA_OPP_POS'])].apply(lambda x: nan_dict[x.SEASON_ID,x.POSITION], axis = 1)

def CM_HEIGHT(x):
    MELO_HT = x['HEIGHT'] * 4.5
    return 'MELO_HT',MELO_HT

def CM_WEIGHT(x):
    MELO_WT = x['WEIGHT'] * 2.0
    return 'MELO_WT',MELO_WT

def CM_CAREER_MINUTES(x):
    MELO_CAREER_MIN = x['SEASON_MIN'] * 2.5
    return 'MELO_CAREER_MIN', MELO_CAREER_MIN

def CM_AGE(x):
    MELO_AGE = x['AGE']
    return 'MELO_AGE', MELO_AGE

def CM_MIN_PER(x):
    MIN_PER = x['MIN'] * 4.5
    return 'MELO_MIN_PER',MIN_PER

def CM_MIN_TOT(x):
    MIN_TOT = (x['MIN'] * x['GP']) * 7
    return 'MELO_MIN_TOT', MIN_TOT

def CM_TRUE_PER(x):
    TRUE_PER = x['TS_PCT'] * 6
    return 'MELO_TRUE_PER',TRUE_PER

def CM_USG_PER(x):
    USG_PER = x['USG_PCT'] * 6
    return 'MELO_USG_PER',USG_PER

def CM_AST_PER(x):
    AST_PER = x['AST_PCT'] * 5
    return 'MELO_AST_PCT', AST_PER

def CM_TO_PER(x):
    TO_PER= x['TM_TOV_PCT'] * 2.5
    return 'MELO_TO_PCT', TO_PER

def CM_REB_PER(x):
    REB_PER = x['REB_PCT'] * 5
    return 'MELO_REB_PCT', REB_PER

def CM_OFF_PM(x):
    OFF_PM= x['OFF_RATING'] * 3
    return 'OFF_PM', OFF_PM

def CM_DF_PM(x):
    DEF_PM= x['DEF_RATING'] * 3
    return 'DEF_PM', DEF_PM

def CM_3FEQ(x):
    MELO_3FEQ = x['3PT_FEQ'] * 3.5
    return 'MELO_3FEQ',MELO_3FEQ

def CM_FT_PER(x):
    MELO_FT_PER = x['FT_PER'] * 3.5
    return 'MELO_FT_PER',MELO_FT_PER

def fab_melo(player, comboMELO):
    root = comboMELO[comboMELO.PLAYER_NAME == player].sort_values('PLAYER_NAME')
    calc_melo_funcs = [CM_WEIGHT, CM_HEIGHT, CM_MIN_PER,CM_CAREER_MINUTES, CM_3FEQ, CM_MIN_TOT, CM_TRUE_PER, CM_USG_PER, CM_AST_PER, CM_TO_PER, CM_REB_PER, CM_OFF_PM, CM_DF_PM,CM_FT_PER,CM_AGE]
    result = root.groupby('SEASON_ID').apply(lambda x: pd.DataFrame(dict([('SEASON_ID',x.SEASON_ID),('PLAYER_NAME',x.PLAYER_NAME)] + map(lambda y: y(x),calc_melo_funcs))))
    return result


def zscore(col):
    return (col - col.mean())/col.std(ddof=0)
    
store_df = []
melo_advanced_df = pd.read_csv("./usage_stats/comboMELO.csv") 
players = set(season_subset(df85_15,1996,2015)['PLAYER_NAME'])
for player in players:
    store_df.append(fab_melo(player,melo_advanced_df))
FAB_MELO = pd.concat(store_df,axis = 0)
melo_cols = ["MELO_MIN_PER", "MELO_MIN_TOT", "DEF_PM","OFF_PM", "MELO_AST_PCT", "MELO_REB_PCT", "MELO_TO_PCT","MELO_USG_PER", "MELO_TRUE_PER","MELO_3FEQ","MELO_FT_PER","MELO_CAREER_MIN","MELO_WT","MELO_HT"]
weights = dict(zip(melo_cols,[4.5,7.0,3.0,3.0,5.0,5.0,2.5,6.0,6.0,3.5,3.5,2.5,2,4.5]))
FAB_MELO[melo_cols] = FAB_MELO[melo_cols].apply(zscore, axis =0)
get_top_ten(FAB_MELO[FAB_MELO.AGE == 34],weights,"Michael Jordan")
len(melo_cols)
