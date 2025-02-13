from pandas import DataFrame

import index_utils as ai
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import common
import index_definition as idef

##以下是否增强信息存疑

def plot_confidence_index(df, gameids, events, time_interval, tfrom=0, to=3600, Display=True) -> DataFrame:
    """
    :return: the index of the attacking team
    """
    detail_sum_df = pd.DataFrame()
    max_cols = 2 #每行最多显示2张
    num_plots = len(gameids)
    rows = (num_plots + max_cols - 1) // max_cols
    cols = min(max_cols, num_plots)
    plt.figure(figsize=(15, rows * 3))  # 设置整个图形的大小，高度根据行数调整
    for ig in range(num_plots):
        gameid = gameids[ig]
        winner_id = common.get_winner(df, gameid)
        ax = plt.subplot(rows, cols, ig + 1)
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
                if (i + 1) == len(num_bins):
                    break
                from_bin = num_bins[i]
                to_bin = num_bins[i + 1]
                aindex, sindex, detail_df = ai.get_attack_index(team_df, gameid, teamid, events, from_bin, to_bin)
                current_sum_confidence += aindex
                current_succ_confidence += sindex
                detail_sum_df = detail_df if detail_sum_df.empty else pd.concat([detail_sum_df, detail_df],ignore_index=True)
                #总体的指数成功率
                current_rate = 0 if current_sum_confidence == 0 else current_succ_confidence / current_sum_confidence
                for gl in goal_times:
                    if from_bin <= gl <= to_bin:
                        goal_x.append(to_bin)
                        goal_sum_y.append(current_sum_confidence)
                        goal_succ_rate_y.append(current_rate)
                confidence_sum.append(current_sum_confidence)
                confidence_rate_sum.append(current_rate)
           # sns.lineplot(x=list(num_bins)[1:], y=confidence_sum,label = teamid,ax=ax)
            sns.lineplot(x=list(num_bins)[1:], y=confidence_rate_sum,label = teamid,ax=ax)
            if teamid == winner_id:
               # ax.text(list(num_bins)[1:][-1], confidence_sum[-1], 'winner', fontsize=12)
                ax.text(list(num_bins)[1:][-1], confidence_rate_sum[-1], 'winner', fontsize=12)

            ax.scatter(x=goal_x, y=goal_succ_rate_y, color='red', marker='o', s=20, label='Goal Point')
            title = 'Game:' + str(gameid) + ' Confidence Index'
            ax.set_title(title)
    plt.tight_layout()
    plt.legend()
    if Display:
        plt.show()
    return detail_sum_df

if __name__ == '__main__':
    data = pd.read_csv("Linhac24-25_Sportlogiq.csv")
    gameids = data['gameid'].unique()
    data['inopponentarea'] = data.apply(common.puck_location, axis=1)
    detail_df = plot_confidence_index(data, gameids[20:40], idef.CONFIDENCE_INDEX,30, 0,3600)
    print()

