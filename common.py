import math
from typing import Tuple, Any, List

import pandas as pd
import numpy as np


def is_first_goal(df,gameid,goal_time):
    goal_actual_index = df[df['compiledgametime'] == goal_time].index




def puck_location(row):
    """
    :param row:
    :return: 1 if the puck is in the home team's half, 0 otherwise
    """
    if row['ishomegame'] == 1 and row['xadjcoord'] >= 0:
        return 0
    elif row['ishomegame'] == 1 and row['xadjcoord'] < 0:
        return 1
    elif row['ishomegame'] == 0 and row['xadjcoord'] <= 0:
        return 1
    elif row['ishomegame'] == 0 and row['xadjcoord'] > 0:
        return 0
    else:
        return math.nan


def get_home_team_id(df, gameid) -> int:
    """
    :return: the home team's id
    """
    return df[(df['ishomegame'] == 1) & (df['gameid'] == gameid)]['teamid'].unique()[0] \
            if gameid is not None else df[df['ishomegame'] == 1]['teamid'].unique()[0]

def get_winner(df, gameid) -> [int,int]:
    """
    :param df:
    :param gameid:
    :return: the winner of the game
    """
    teams = df[(df['gameid'] == gameid)]['teamid'].unique()
    last_row = df[(df['gameid'] == gameid)].iloc[-1]
    last_row_team_id = last_row['teamid']
    another_team_id = teams[0] if teams[0] != last_row_team_id else teams[1]
    last_row_score_diff = last_row['scoredifferential']
    if last_row_score_diff == 0:
        return [last_row_team_id, another_team_id, False]
    return [last_row_team_id,another_team_id,True] if last_row_score_diff > 0 else [another_team_id,last_row_team_id,True]




# 控球率
# 最大控球时长
def get_control_rate(df, gameid,tfrom = 0,to = 3600) -> tuple[float | Any, float | Any, list[Any], list[Any]]:
    df_copy = df.copy()
    df_copy = df_copy[(df_copy['gameid'] == gameid) & (df_copy['compiledgametime'] >= tfrom) & (df_copy['compiledgametime'] <= to)]
    home_possession_time = 0
    home_possession_period = list()
    visit_possession_time = 0
    visit_possession_period = list()
    home_team_id = get_home_team_id(df_copy, gameid)
    mask = df_copy['teaminpossession'].isnull()
    groups = mask.cumsum()
    filtered_df = df_copy[~mask]
    filtered_groups = groups[~mask]
    adjusted_groups = filtered_groups - filtered_groups.min()
    grouped = filtered_df.groupby(adjusted_groups)
    for group_id, group_data in grouped:
        hold_time = group_data['compiledgametime'].max() - group_data['compiledgametime'].min()
        current_team_id = group_data['teaminpossession'].iloc[0]
        if current_team_id == home_team_id:
            home_possession_time += hold_time
            home_possession_period.append(hold_time)
            # home_max_possession_time = hold_time if hold_time > home_max_possession_time else home_max_possession_time
        else:
            visit_possession_time += hold_time
            visit_possession_period.append(hold_time)
    #         visit_max_possession_time = hold_time if hold_time > visit_max_possession_time else visit_max_possession_time
    return (home_possession_time / (home_possession_time + visit_possession_time)),\
             (visit_possession_time / (home_possession_time + visit_possession_time)), home_possession_period, visit_possession_period




def get_control_rate0(df, gameid,tfrom = 0, to= 3600,home_team_id = None) -> tuple[float | Any, float | Any, list[Any], list[Any]]:

    home_possession_time = 0
    home_possession_period = list()
    visit_possession_time = 0
    visit_possession_period = list()
    df_copy = df.copy()
    df_copy = df_copy[(df_copy['gameid'] == gameid) & (df_copy['compiledgametime'] >= tfrom) & (df_copy['compiledgametime'] <= to)]
    if home_team_id is None:
        home_team_id = get_home_team_id(df_copy, None)
    #去掉currentinpossession为空的行
    df_copy =  df_copy.dropna(subset=['currentpossession'])
    grouped = df_copy.groupby("currentpossession")
    for group_id, group_data in grouped:
        hold_time = group_data['compiledgametime'].max() - group_data['compiledgametime'].min()
        current_team_id = group_data['teaminpossession'].iloc[0]
        if len(group_data) == 0:
            continue
        if current_team_id == home_team_id:
            home_possession_time += hold_time
            home_possession_period.append(hold_time)
            # home_max_possession_time = hold_time if hold_time > home_max_possession_time else home_max_possession_time
        else:
            visit_possession_time += hold_time
            visit_possession_period.append(hold_time)
    home_possession_rate = 0 if (home_possession_time + visit_possession_time) == 0 else (home_possession_time / (home_possession_time + visit_possession_time))
    visit_possession_rate = 0 if (home_possession_time + visit_possession_time) == 0 else (visit_possession_time / (home_possession_time + visit_possession_time))
    return home_possession_rate,visit_possession_rate, home_possession_period, visit_possession_period

def get_carry_time(df, gameid, tfrom = 0, to = 3600):
    """
    获取每轮的进攻时间
    :param df:`
    :param gameid:
    :param tfrom:
    :param to:
    :return:
    """
    all_carry_events = []
    df_copy = df.copy()
    df_copy = df_copy[(df_copy['gameid'] == gameid) & (df_copy['compiledgametime'] > tfrom) & (df_copy['compiledgametime'] <= to)]
    grouped = df_copy.groupby(['gameid','currentpossession'])
    def consume_carry_time(row,possession_time_list,carry_event_list,carry_start_player):
        current_player = row['playerid']
        current_time = row['compiledgametime']
        event_name = row['eventname']
        if current_player != carry_start_player: return
        else:
            possession_time_list.append(current_time)
            carry_event_list.append(event_name)

    carry_time = 0
    for gameid, group_data in grouped:
        event_names = group_data['eventname'].unique()
        if "carry" not in event_names:continue
        carry_start_index = group_data[group_data['eventname']=='carry'].index.tolist()[0]
        carry_start_player = group_data.loc[carry_start_index,'playerid']
        carry_start_time = group_data.loc[carry_start_index,'compiledgametime']
        possession_time_list = [carry_start_time]
        carry_event_list = []
        carry_df = group_data.loc[carry_start_index:]
        carry_df.apply(consume_carry_time,axis=1,args=(possession_time_list,carry_event_list,carry_start_player))
        carry_time += possession_time_list[-1] - possession_time_list[0]
        all_carry_events.append((carry_start_time,carry_event_list))






if __name__ == '__main__':
    data = pd.read_csv("data/Linhac24-25_Sportlogiq.csv")

   # home_control_rate, visit_control_rate , home_max_control_time , visit_max_control_time = get_control_rate(data, 64485)

    # home_control_rate, visit_control_rate , home_max_control_time , visit_max_control_time = get_each_round_time(data, 64485)
    #
    # print(home_control_rate, visit_control_rate, home_max_control_time, visit_max_control_time)
    get_carry_time(data,72393,0,3600)