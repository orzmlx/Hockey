import pandas as pd
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
def get_index(df,gameid,teamid,events ,tfrom=0 ,to=3600 ):
    """
    :return: the index of the attacking team
    """

    succ_index = 0
    index = 0
    df_copy = df.copy()
    if gameid is not None :
        df_copy = df_copy[df_copy['gameid'] == gameid]
    if teamid is not None:
        df_copy = df_copy[df_copy['teamid'] == teamid]
    df_copy = df_copy[(df_copy['compiledgametime'] > tfrom) & (df_copy['compiledgametime'] <= to)]
    score_diff_min = df_copy['scoredifferential'].min()
    score_diff_max = df_copy['scoredifferential'].max()
    score_diff = score_diff_min if abs(score_diff_min) > abs(score_diff_max) else score_diff_max
    single_index_dict = {"teamid":teamid,"gameid":gameid,"elapsed_sec":to,"score_diff":score_diff}
    for name,action in events.items():
        #logging.info(f"the number of {action} ）")
        condition = df_copy.eval(action)
        result = df_copy[condition]
        success_result = result[result['outcome'] == 'successful']
        succ_index = succ_index + len(success_result) #计算成功总数量,包括所有的event
        index = index + len(result) #计算总数量
        single_index_dict = {**single_index_dict, **{name: len(result)},
                             **{name + "_succ": len(success_result)}}
    # index_dict = {**single_index_dict, **{"elapsed_sec": to}, **{"score_diff": score_diff}}
        #logging.info(f"the number of {action} is {len(result)}.total number of index is {index}")
    return index, succ_index,pd.DataFrame([single_index_dict])

def get_score_time(df,gameid,teamid):
    df_copy = df.copy()
    if gameid is not None :
        df_copy = df_copy[df_copy['gameid'] == gameid]
    if teamid is not None:
        df_copy = df_copy[df_copy['teamid'] == teamid]
    goals = df_copy[(df_copy['eventname'] == 'goal') & (df_copy['outcome'] == 'successful')]
    return  goals['compiledgametime'].values




def plot_index(df, gameid, teamid, events, time_interval, tfrom=0, to=3600) -> int:
    """
    :return: the index of the attacking team
    """
    team_df = df[(df['gameid'] == gameid) & (df['teamid'] == teamid)]
    goal_times = get_score_time(team_df, gameid, teamid)
    goal_y = []
    goal_x = []
    num_bins = np.arange(tfrom, to + time_interval, time_interval)
    event_num = []
    for i in range(len(num_bins)):
        if (i + 1) == len(num_bins):
            break
        from_bin = num_bins[i]
        to_bin = num_bins[i + 1]
        aindex,sndex = get_index(team_df, gameid, teamid, events, from_bin, to_bin)
        for gl in goal_times:
            if from_bin <= gl <= to_bin:
                goal_x.append(to_bin)
                goal_y.append(aindex)
        event_num.append(aindex)
    sns.lineplot(x=list(num_bins)[1:], y=event_num)

    plt.scatter(x=goal_x, y=goal_y, color='red', marker='o', s=20, label='Special Point')
    plt.show()
    return event_num