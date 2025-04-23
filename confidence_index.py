from pandas import DataFrame
import index_utils as ai
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import common
import index_definition as idef
from tqdm import tqdm
import math
##以下是否增强信息存疑

custom_style = {
    "desc": "处理中",          # 进度条前的描述文字（支持Emoji）
    "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    "ncols": 100,                # 进度条总宽度
  #  "colour": "blue",         # 进度条颜色（支持hex颜色码）
    "ascii": " ░▓",              # 自定义进度字符（空槽和填充字符）
    "dynamic_ncols": True        # 根据终端宽度自动调整
}


def get_index_cut_by_time(df, gameids, events, time_interval, tfrom=0, to=3600) -> DataFrame:
    """
    :return: the index of the attacking team
    """
    if to - tfrom < time_interval:
        raise ValueError("The time interval is too large")
    detail_sum_df = pd.DataFrame()
    # 设置整个图形的大小，高度根据行数调整
    with tqdm(total=len(gameids),colour='BLUE', dynamic_ncols=True,desc="🚀 Processing") as pbar:
        for ig in range(len(gameids)):  # max_cols = 2 #每行最多显示2张
            current_batch_size = math.floor((ig / len(gameids)) * 100) #向下取整计算进度
            gameid = gameids[ig]
            team_df = df[(df['gameid'] == gameid)]
            teams = team_df['teamid'].unique()
            for teamid in teams:
                goal_times = ai.get_score_time(team_df, gameid, teamid)
                goal_sum_y = []
                goal_succ_rate_y = []
                goal_x = []
                num_bins = np.arange(tfrom, to + time_interval, time_interval)
                confidence_sum = []
                confidence_rate_sum = []
                current_sum_confidence = 0
                current_succ_confidence = 0
                for i in range(len(num_bins)):
                    if (i + 1) == len(num_bins):break
                    from_bin = num_bins[i]
                    to_bin = num_bins[i + 1]
                    aindex, sindex, detail_df = ai.get_index(team_df, gameid, teamid, events, from_bin, to_bin)
                    current_sum_confidence += aindex
                    current_succ_confidence += sindex
                    #总体的指数成功率
                    current_rate = 0 if current_sum_confidence == 0 else current_succ_confidence / current_sum_confidence
                    for gl in goal_times:
                        if from_bin <= gl <= to_bin:
                            goal_x.append(to_bin)
                            goal_sum_y.append(current_sum_confidence)
                            goal_succ_rate_y.append(current_rate)
                    confidence_sum.append(current_sum_confidence)
                    confidence_rate_sum.append(current_rate)
                    detail_sum_df = detail_df if detail_sum_df.empty else pd.concat([detail_sum_df, detail_df],ignore_index=True)
            pbar.update(current_batch_size)
    return detail_sum_df

if __name__ == '__main__':
    data = pd.read_csv("data/Linhac24-25_Sportlogiq.csv")
    gameids = data['gameid'].unique()
    data['inopponentarea'] = data.apply(common.puck_location, axis=1)
    detail_df = get_index_cut_by_time(data, gameids[20:40], idef.CONFIDENCE_INDEX,30, 0,3600)
    print()

