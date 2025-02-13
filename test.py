import pandas as pd
import numpy as np
from river import linear_model, preprocessing, metrics, stream,optim
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


# ========================================
# 1. 生成模拟实时比赛数据流
# ========================================
def generate_live_match_data(num_games=5, updates_per_game=10):
    """
    生成实时比赛数据流，模拟多场比赛的实时更新
    - num_games: 模拟的比赛场次
    - updates_per_game: 每场比赛的数据更新次数
    """
    np.random.seed(42)
    data = []

    for game_id in range(num_games):
        # 生成比赛基本信息
        start_time = datetime.now() - timedelta(minutes=48)  # 假设比赛时长48分钟
        final_score_diff = np.random.randint(-15, 15)  # 最终分差（主队 - 客队）
        result = 1 if final_score_diff > 0 else 0  # 1=主队胜，0=客队胜

        # 生成实时时间序列数据
        for step in range(updates_per_game):
            elapsed_time = (step + 1) * 4.8  # 模拟每4.8分钟更新一次数据（共10次）
            current_time = start_time + timedelta(minutes=elapsed_time)

            # 生成动态分差（加入随机波动）
            progress = elapsed_time / 48  # 比赛进度（0~1）
            current_diff = final_score_diff * (0.3 + 0.7 * progress) + np.random.normal(0, 3)

            data.append([
                game_id,
                current_time,
                elapsed_time,
                current_diff,
                result
            ])

    df = pd.DataFrame(data, columns=[
        'game_id', 'timestamp', 'elapsed_minutes',
        'score_diff', 'result'
    ])
    return df.sort_values('timestamp')


# 生成5场比赛的实时数据流（每场10次更新）
live_data = generate_live_match_data()
print("实时数据样例:")
print(live_data.head(3))

# ========================================
# 2. 初始化在线学习模型
# ========================================
# 特征工程：标准化处理
scaler = preprocessing.StandardScaler()

# 使用逻辑回归模型（适合概率预测）
model = linear_model.LogisticRegression(
    optimizer=optim.SGD(0.01),  # 随机梯度下降
    #loss=metrics.LogLoss()  # 对数损失
)

# 指标跟踪
metric = metrics.ROCAUC()
win_rates = []

# ========================================
# 3. 模拟实时数据流处理
# ========================================
print("\n开始实时更新...")
for i, (X, y) in enumerate(stream.iter_pandas(
        live_data[['elapsed_minutes', 'score_diff']],
        live_data['result'])):
    # 特征标准化（逐步更新scaler）
    scaler.learn_one(X)
    X_scaled = scaler.transform_one(X)
    # 使用当前模型预测胜率
    y_pred = model.predict_proba_one(X_scaled)
    win_prob = y_pred.get(True, 0.5)  # 获取主队胜率

    # 更新模型
    model.learn_one(X_scaled, y)

    # 记录结果
    win_rates.append({
        'timestamp': live_data.iloc[i]['timestamp'],
        'true_result': y,
        'pred_win_rate': win_prob,
        'score_diff': X['score_diff']
    })

    # 打印实时日志
    if (i + 1) % 10 == 0:
        print(f"已处理 {i + 1} 条数据 | "
              f"当前分差: {X['score_diff']:.1f} | "
              f"预测胜率: {win_prob:.2%} | "
              f"真实结果: {'主队胜' if y == 1 else '客队胜'}")

# 转换为DataFrame
results_df = pd.DataFrame(win_rates)

# ========================================
# 4. 可视化胜率变化
# ========================================
plt.figure(figsize=(12, 6))

# 绘制胜率曲线
plt.plot(results_df['timestamp'], results_df['pred_win_rate'],
         label='预测胜率', color='#2c7bb6', linewidth=2)

# 标记真实比赛结果
for game_id in live_data['game_id'].unique():
    game_data = live_data[live_data['game_id'] == game_id]
    result = '主队胜' if game_data['result'].iloc[0] == 1 else '客队胜'
    plt.scatter(
        game_data['timestamp'].iloc[-1],  # 比赛结束时间
        game_data['result'].iloc[-1],  # 最终结果
        marker='*' if result == '主队胜' else 'X',
        s=150,
        label=f'比赛{game_id}结果: {result}'
    )

# 标注关键分差
for idx, row in results_df.iterrows():
    if abs(row['score_diff']) > 10:
        plt.annotate(
            f"{row['score_diff']:.1f}",
            (row['timestamp'], row['pred_win_rate']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )

plt.title('实时胜率预测动态更新')
plt.xlabel('时间')
plt.ylabel('主队胜率')
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()