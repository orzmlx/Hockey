import numpy as np
import index_definition
import confidence_index as ci
import common
import pandas as pd
def feature_diff(f1,f2):
    if type(f1) is pd.Series and type(f2) is pd.Series:
        f1 = f1.reset_index(drop=True)
        f2= f2.reset_index(drop=True)
        return  (f1- f2).tolist()
    return f1 - f2



def feature_ratio(f1,f2,max_ratio):
    if type(f1) is pd.Series and type(f2) is pd.Series:
        f1 = f1.reset_index(drop=True)
        f2 = f2.reset_index(drop=True)
        ratio = f1 / (f2 + 1e-6)
        return  (ratio.clip(upper=max_ratio)).tolist()
        #return np.log1p(ratio).tolist()
    return np.log1p(f1 / (f2  + 1e-6))



def relative_feature_extract(colname,max_ratio = 10):
    # if 'rate' not in colname:
    #     return lambda x, y: feature_diff(x, y)
    # else:
    #     return lambda x, y: feature_ratio(x, y)
    return lambda x, y: feature_ratio(x, y,max_ratio)

def extract_sum_feature(df,suffix):

    df.iloc[0] = df.sum()
    df = df.add_suffix(suffix)
    # 仅保留第一行
    return df.iloc[[0]]

def extract_features(data , df, suffix, events,tfrom,to):
    event_names = list(events.keys())
    event_names = [str(item) + '_succ' for item in event_names] + event_names
    grouped = df.groupby(['gameid','teamid'])
    res = pd.DataFrame()
    for name, group in grouped:
        gameid = name[0]
        teamid = name[1]
        home_team_id = common.get_home_team_id(data, gameid)
        winner = common.get_winner(data,gameid)
        team_df = group[event_names]
        # team_df.iloc[0] = team_df.sum()
        # team_df = team_df.add_suffix(suffix)
        # # 仅保留第一行
        # team_df = team_df.iloc[[0]]
        team_df = extract_sum_feature(team_df, suffix)
        team_df['gameid'] = gameid
        team_df['teamid'] = teamid
        team_df['score_diff' + suffix] = group.iloc[-1]['score_diff']
        team_df['is_home'] = 1 if teamid == home_team_id else 0
        #获取这段时间的控球率
        home_control_rate, visit_control_rate, home_possession_time, visit_possession_time =common.get_control_rate0(data, gameid, tfrom, to)
        team_df['control_rate' + suffix] = home_control_rate if teamid == home_team_id else visit_control_rate
        team_df['control_time' + suffix] = sum(home_possession_time) if teamid == home_team_id else sum(visit_possession_time)
        team_df['win'] = 1 if winner[2] is False else 0 if winner[0] == teamid else 1
        res = team_df if res.empty else pd.concat([res, team_df], axis=0)
    return res


def extract_relative_features(data , df, suffix, events,tfrom,to):
    event_names = list(events.keys())
    res = pd.DataFrame()
    event_names = [str(item) + '_succ' for item in event_names] + event_names
    #使用一场比赛的两个队伍的数据对比，计算相对值
    grouped = df.groupby(['gameid'])
    exclude_feature = ['teamid','gameid','elapsed_sec','score_diff']
    for name, group in grouped:
        gameid = name
        teamids = group['teamid'].unique()
        home_team_id = common.get_home_team_id(data, gameid)
        winner = common.get_winner(data,gameid)
        home_control_rate, visit_control_rate, home_possession_time, visit_possession_time =common.get_control_rate0(data, gameid, tfrom, to)
        relative_control_rate =feature_ratio(home_control_rate, visit_control_rate,10)
        relative_control_time = feature_diff(sum(home_possession_time) , sum(visit_possession_time))
        home_df = group[group['teamid'] == home_team_id]
        visit_df = group[group['teamid'] == [x for x in teamids if x != home_team_id][0]]
        home_df_copy =  home_df.copy()
        for col in visit_df.columns.tolist():
            if col in exclude_feature:
                continue
            home_df_copy[col] =relative_feature_extract(col)(home_df[col],visit_df[col])
        home_df_copy.drop(columns=['gameid', 'teamid', 'elapsed_sec'], inplace=True, errors='ignore')
        home_df_copy = extract_sum_feature(home_df_copy, suffix)
        home_df_copy['gameid'] = gameid
        home_df_copy['score_diff' + suffix] = home_df.iloc[-1]['score_diff']
        home_df_copy['control_rate' + suffix] = relative_control_rate
        home_df_copy['control_time' + suffix] = relative_control_time
        home_df_copy['win'] = 0 if winner[2] is False else 1 if winner[0] == home_team_id else 0

        # score diff是主场队伍的
        res = pd.concat([res, home_df_copy], axis=0)
        #extract_sum_feature(data, home_df_copy, suffix, group, home_team_id)
    return res

def enhance_features():
    """
    特征扰动 + 交换主客场 → 变成 300~750 场比赛的数据,解决训练样本过少的问题
    :return: 
    """
    pass

def exchange_visit_home(data, df):
    pass


def add_noise():
    pass


if __name__ == '__main__':
    data = pd.read_csv("data/Linhac24-25_Sportlogiq.csv")
    data['inopponentarea'] = data.apply(common.puck_location, axis=1)
    unique_gameids = data['gameid'].unique()
    gameids = unique_gameids.tolist()[1:5]
    all_events_index = {**index_definition.EXERTION_INDEX, **index_definition.CONFIDENCE_INDEX}
    #将第一节时间也分成两个部分,一个是0到540秒，一个是540到1200秒

    first_period_index_df1 =ci.get_index_cut_by_time(data, gameids, all_events_index, 180,tfrom = 0, to = 540)
    res1 = extract_relative_features(data, first_period_index_df1, "_p1", all_events_index, 0, 540)
    #res1 = extract_features(data, first_period_index_df1, "_p1", all_events_index, 0, 540)
    # print()