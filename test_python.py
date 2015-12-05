import numpy as np

def sigmoidfun(x):
	return 1/(1+np.exp(-0.007*(x-800)))

merged_df = pd.merge(df85_15,df85_15.groupby(["PLAYER_NAME",'OPP']).apply(lambda x: x.FANTASY_PTS).reset_index())#.rename(columns = {0:"FANTASY_PTS_AGAINST"}), left_on=['PLAYER_NAME','OPP'],right_on=['PLAYER_NAME','OPP'])

high_rollers = testdata.groupby(["PLAYER_NAME",'OPP']).apply(lambda x: rolling_cols(x,['FANTASY_PTS'],3,'mean'))
