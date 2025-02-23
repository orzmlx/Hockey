import index_definition
import confidence_index as ci
import common
import pandas as pd
def feature_diff(f1,f2):
    return f1 - f2



def feature_ratio(f1,f2):
    return f1 / (f2  + 1e-6)



def relative_feature_extract(colname):
    if 'rate' in colname:
        return lambda x, y: feature_diff(x, y)
    else:
        return lambda x, y: feature_ratio(x, y)


def extract_sum_feature(df,gameid,teamid,suffix,group,home_team_id):
    df.iloc[0] = df.sum()
    df = df.add_suffix(suffix)
    # 仅保留第一行
    df = df.iloc[[0]]
    df['gameid'] = gameid
    df['teamid'] = teamid
    df['score_diff' + suffix] = group.iloc[-1]['score_diff']
    df['is_home'] = 1 if teamid == home_team_id else 0

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
        team_df.iloc[0] = team_df.sum()
        team_df = team_df.add_suffix(suffix)
        # 仅保留第一行
        team_df = team_df.iloc[[0]]
        team_df['gameid'] = gameid
        team_df['teamid'] = teamid
        team_df['score_diff' + suffix] = group.iloc[-1]['score_diff']
        team_df['is_home'] = 1 if teamid == home_team_id else 0
        #获取这段时间的控球率
        home_control_rate, visit_control_rate, home_possession_time, visit_possession_time =common.get_control_rate0(data, gameid, tfrom, to)
        team_df['control_rate' + suffix] = home_control_rate if teamid == home_team_id else visit_control_rate
        team_df['control_time' + suffix] = sum(home_possession_time) if teamid == home_team_id else sum(visit_possession_time)
        team_df['win'+ suffix] = 1 if winner[2] is False else 0 if winner[0] == teamid else 1
        res = team_df if res.empty else pd.concat([res, team_df], axis=0)
    return res


def extract_relative_feature(data , df, suffix, events,tfrom,to):
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
        home_df= group[group['teamid'] == home_team_id]
        visit_df = group[group['teamid'] == [x for x in teamids if x != home_team_id][0]]
        home_df_copy =  home_df.copy()
        for col in visit_df.columns.tolist():
            if col in exclude_feature:
                continue
            home_df_copy[col] =relative_feature_extract(col)(home_df[col],visit_df[col])
        #extract_sum_feature(data, home_df_copy, suffix, group, home_team_id)
    return res


if __name__ == '__main__':
    data = pd.read_csv("Linhac24-25_Sportlogiq.csv")
    data['inopponentarea'] = data.apply(common.puck_location, axis=1)
    unique_gameids = data['gameid'].unique()
    gameids = unique_gameids.tolist()[1:2]
    all_events_index = {**index_definition.EXERTION_INDEX, **index_definition.CONFIDENCE_INDEX}
    #将第一节时间也分成两个部分,一个是0到540秒，一个是540到1200秒

    first_period_index_df1 =ci.get_index_cut_by_time(data, gameids, all_events_index, 180,tfrom = 0, to = 540)
    res1 = extract_relative_feature(data, first_period_index_df1, "_p1", all_events_index, 0, 540)
