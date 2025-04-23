from itertools import accumulate

import seaborn as sns
import common
import index_definition as idef
import confidence_index as ci
import pandas as pd
from river import linear_model, preprocessing, metrics, stream, optim
import math
import dill
import matplotlib.pyplot as plt
from itertools import zip_longest

from common import get_winner

#低通滤波器，实现胜率曲线平滑
class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None

    def update(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * value + (1 - self.alpha) * self.value
        return self.value

class DataAccumulator:
    def __init__(self,data):
        self.data = data
        self.prev_row = None

    def get(self,row):
        if self.prev_row is None:
            #保存当前行
            self.prev_row = row._asdict()
            return  {
                        'elapsed_sec': row.elapsed_sec,
                        'score_diff': row.score_diff,
                        "goal_succ": row.goal_succ,
                        "accuracy_reception_succ": row.accuracy_reception_succ,
                        "accuracy_pass_succ": row.accuracy_pass_succ,
                        "efficient_block_succ": row.efficient_block_succ,
                        "body_check_succ": row.body_check_succ,
                }
        else:
            goal_succ =  int(row.goal_succ + self.prev_row['goal_succ'])
            accuracy_reception_succ =  int(row.accuracy_reception_succ +  self.prev_row['accuracy_reception_succ'])
            accuracy_pass_succ =  int(row.accuracy_pass_succ +  self.prev_row['accuracy_pass_succ'])
            efficient_block_succ =   int(row.efficient_block_succ +  self.prev_row['efficient_block_succ'])
            body_check_succ =  int(row.body_check_succ +  self.prev_row['body_check_succ'])
            # }
            X = {
                'elapsed_sec': row.elapsed_sec,
                'score_diff': row.score_diff,
                "goal": goal_succ,
                "accuracy_reception_succ": accuracy_reception_succ,
                "accuracy_pass_succ": accuracy_pass_succ,
                "efficient_block_succ": efficient_block_succ,
                "body_check_succ": body_check_succ,
            }
            #更新当前行
            self.prev_row['score_diff'] = row.score_diff
            self.prev_row['elapsed_sec'] = row.elapsed_sec
            self.prev_row['goal_succ'] = goal_succ
            self.prev_row['accuracy_reception_succ'] = accuracy_reception_succ
            self.prev_row['accuracy_pass_succ'] = accuracy_pass_succ
            self.prev_row['efficient_block_succ'] = efficient_block_succ
            self.prev_row['body_check_succ'] = body_check_succ

            # self.prev_row.score_diff = row.score_diff
            # self.prev_row.elapsed_sec = row.elapsed_sec
            # self.prev_row.goal_succ = goal_succ
            # self.prev_row.accuracy_reception_succ = accuracy_reception_succ
            # self.prev_row.accuracy_pass_succ = accuracy_pass_succ
            # self.prev_row.efficient_block_succ = efficient_block_succ
            # self.prev_row.body_check_succ = body_check_succ
            return X

scaler =  preprocessing.StandardScaler()


def generate_live_match_data(gameids, updates_per_game=10):
    """

    """
    data = pd.read_csv("../data/Linhac24-25_Sportlogiq.csv")
    # gameids = data['gameid'].unique()
    data['inopponentarea'] = data.apply(common.puck_location, axis=1)
    detail_df = ci.get_confidence_index(data, gameids, idef.CONFIDENCE_INDEX, 15, 0, 3600,Display=False)
    return detail_df



def train(train_data,df):
    # 维护每场比赛的状态 {game_id: MatchProcessor}
    game_processors = {}
    win_rates = []

    accuracy = metrics.Accuracy()
    roc_auc = metrics.ROCAUC()
    # 使用逻辑回归模型（适合概率预测）
    model = linear_model.LogisticRegression(
        optimizer=optim.SGD(0.01),  # 随机梯度下降
        # loss=metrics.LogLoss()  # 对数损失
    )
    gameids = train_data['gameid'].unique()
    row_index = 0
    for gameid in gameids:
        winner = common.get_winner(df, gameid)
        y = 2 if winner[2] is False else None
        winnerid = winner[0]
        game_data = train_data[train_data['gameid'] == gameid]
        teamids = train_data['teamid'].unique()
        for teamid in teamids:
            team_data = game_data[game_data['teamid'] == teamid]
            team_accumulators = DataAccumulator(team_data)
            for row in team_data.itertuples():
                row_index = row_index + 1
                X = team_accumulators.get(row)
                if y is None:
                    y = 1 if teamid == winnerid else 0 #胜利者为1，失败者为0
                scaler.learn_one(X)
                X_scaled = scaler.transform_one(X)
                # 使用当前模型预测胜率
                y_pred = model.predict_proba_one(X_scaled)
                win_prob = y_pred.get(True, 0.5)  # 获取主队胜率
                # 更新评估指标
                accuracy.update(y, win_prob > 0.5)
                roc_auc.update(y, win_prob)
                # 更新模型
                model.learn_one(X_scaled, y)
                # 记录结果
                win_rates.append({
                    'gameid': gameid,
                    'teamid': teamid,
                    'elapsed_sec': row.elapsed_sec,
                    'true_result': y,
                    'pred_win_rate': win_prob,
                    'score_diff': X['score_diff']
                })
                if (row_index + 1) % 10 == 0:
                    print(f"已处理 {row_index} 条数据 | "
                          f"比赛 {gameid} | "
                          f"队伍 {teamid} | "
                          f"胜率 {win_prob} | "
                          f"准确率: {accuracy.get():.2%} | "
                          f"AUC: {roc_auc.get():.2f}")
    return model

def predict(test_data,gameid,teamid, winnerid, model_dict):
    test_accuracy = metrics.Accuracy()
    test_roc_auc = metrics.ROCAUC()
    home_win_rates = []
    away_win_rates = []
    filter = LowPassFilter(0.2)
    test_data = test_data[(test_data['gameid'] == gameid) & (test_data['teamid'] == teamid)]
    accumulate = DataAccumulator(test_data)
    index = 0
    for row in test_data.itertuples():
        y = 1 if teamid == winnerid else 0
        index = index + 1
        X = accumulate.get(row)
        scaler = model_dict['scaler']
        model = model_dict['model']
        X_scaled = scaler.transform_one(X)  # 使用训练集的scaler
        y_pred = model.predict_proba_one(X_scaled)
        win_rate = filter.update(y_pred[True])
        home_win_rates.append(win_rate)
        away_win_rates.append(1- win_rate)
        test_accuracy.update(y, y_pred[True] > 0.5)
        test_roc_auc.update(y, y_pred)
    print(f"[测试集] 准确率: {test_accuracy.get():.2%} | AUC: {test_roc_auc.get():.2f}")


    return pd.DataFrame(list(zip_longest(list(test_data['elapsed_sec']),home_win_rates, away_win_rates,list(test_data['goal_succ']))),
                        columns=['elapsed_sec','home_win_rate','away_win_rate','goal'])
    #sns.lineplot(x=list(test_data['elapsed_sec']), y=test_win_rates, label=teamid)
    #plt.show()
    # sns.plot(
    #     test_data['elapsed_sec'],
    #     test_win_rates,
    #     label=f'比赛 {gameid} | 队伍 {teamid} | 胜利者 {winnerid}',
    #     marker='o'
    # )


if __name__ == '__main__':
    data = pd.read_csv("../data/Linhac24-25_Sportlogiq.csv")
    gameids = data['gameid'].unique()
    train_data = generate_live_match_data(gameids[31:156])
    #缺少的score_diff值用上面一行填充
    train_data['score_diff'].fillna(method='pad', inplace=True)
    model = train(train_data, data)
    save_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': ['elapsed_sec', 'score_diff', 'goal', 'accuracy_reception', 'accuracy_pass',
                          'efficient_block', 'body_check']  # 记录特征顺序
    }
    with open('trained_model.pkl', 'wb') as f:
       dill.dump(save_data, f)
    # test_data = generate_live_match_data(gameids[10:30])
    # test_data['score_diff'].fillna(method='pad', inplace=True)
    # with open('trained_model.pkl', 'rb') as f:
    #     model = dill.load(f)
    # winner = get_winner(data,68819)[0]
    # predict(test_data,68819,855,winner, model)
    # print()
