import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# CSVファイルを読み込み
print("CSVファイルを読み込み中...")
df = pd.read_csv('usdjpy_1min.csv', sep=';', header=None, 
                 names=['datetime', 'O', 'H', 'L', 'C', 'volume'])

# 日時列をdatetime型に変換
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S')
df = df.sort_values('datetime')

# 最新2日分のデータを抽出
latest_date = df['datetime'].max()
two_days_ago = latest_date - timedelta(days=2)
df_recent = df[df['datetime'] >= two_days_ago].copy()

print(f"データ期間: {df_recent['datetime'].min()} から {df_recent['datetime'].max()}")
print(f"データ数: {len(df_recent)} 件")

# 時系列インデックスを作成（分単位）
df_recent = df_recent.reset_index(drop=True)
df_recent['time_index'] = range(len(df_recent))

# 終値（C）を使用してOLS回帰
X = sm.add_constant(df_recent['time_index'])
model = sm.OLS(df_recent['C'], X)
results = model.fit()

# 予測値と残差を計算
df_recent['y_hat'] = results.fittedvalues
df_recent['resid'] = df_recent['C'] - df_recent['y_hat']

# 標準偏差を計算
sigma = df_recent['resid'].std()
k = 2  # 標準偏差倍率

# 上下バンドを計算
df_recent['upper'] = df_recent['y_hat'] + k * sigma
df_recent['lower'] = df_recent['y_hat'] - k * sigma

# プロット
plt.figure(figsize=(15, 8))

# 価格データをプロット
plt.plot(df_recent['datetime'], df_recent['C'], label='USD/JPY Close', 
         color='gray', alpha=0.7, linewidth=0.8)

# OLS中心線
plt.plot(df_recent['datetime'], df_recent['y_hat'], label='OLS Center', 
         color='blue', linewidth=2)

# 上下バンド
plt.plot(df_recent['datetime'], df_recent['upper'], 
         label=f'+{k}σ Band', color='red', linestyle='--', linewidth=1.5)
plt.plot(df_recent['datetime'], df_recent['lower'], 
         label=f'-{k}σ Band', color='green', linestyle='--', linewidth=1.5)

# バンド領域を塗りつぶし
plt.fill_between(df_recent['datetime'], df_recent['upper'], df_recent['lower'], 
                 color='orange', alpha=0.1)

# グラフの設定
plt.title(f'USD/JPY OLS Regression Channel (Last 2 Days)\n'
          f'Period: {df_recent["datetime"].min().strftime("%Y-%m-%d %H:%M")} - '
          f'{df_recent["datetime"].max().strftime("%Y-%m-%d %H:%M")}', 
          fontsize=14, fontweight='bold')

plt.xlabel('Date/Time', fontsize=12)
plt.ylabel('USD/JPY Price', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# x軸の日時フォーマット
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 統計情報を表示
print(f"\n=== OLS回帰統計 ===")
print(f"R-squared: {results.rsquared:.4f}")
print(f"標準偏差 (σ): {sigma:.4f}")
print(f"チャネル幅: ±{k*sigma:.4f}")
print(f"中心線の傾き: {results.params[1]:.6f}")
print(f"中心線の切片: {results.params[0]:.4f}") 