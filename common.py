import math
from typing import Tuple, Any, List

import pandas as pd
import numpy as np

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

def get_winner(df, gameid) -> int:
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
    return last_row_team_id if last_row_score_diff > 0 else another_team_id


# 控球率
# 最大控球时长
def get_control_rate(df, gameid) -> tuple[float | Any, float | Any, list[Any], list[Any]]:
    df_copy = df.copy()
    df_copy = df_copy[df_copy['gameid'] == gameid]
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




def get_control_rate0(df, gameid) -> tuple[float | Any, float | Any, list[Any], list[Any]]:

    home_possession_time = 0
    home_possession_period = list()
    visit_possession_time = 0
    visit_possession_period = list()
    df_copy = df.copy()
    df_copy = df_copy[df_copy['gameid'] == gameid]
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
    return (home_possession_time / (home_possession_time + visit_possession_time)),\
             (visit_possession_time / (home_possession_time + visit_possession_time)), home_possession_period, visit_possession_period
if __name__ == '__main__':
    data = pd.read_csv("Linhac24-25_Sportlogiq.csv")

   # home_control_rate, visit_control_rate , home_max_control_time , visit_max_control_time = get_control_rate(data, 64485)

    home_control_rate, visit_control_rate , home_max_control_time , visit_max_control_time = get_each_round_time(data, 64485)

    print(home_control_rate, visit_control_rate, home_max_control_time, visit_max_control_time)
