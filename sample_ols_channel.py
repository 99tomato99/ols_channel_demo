import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ダミーの1分足データ生成
np.random.seed(42)
n = 200
time = np.arange(n)
price = 100 + 0.05 * time + np.random.normal(0, 1, n)
df = pd.DataFrame({'time': time, 'price': price})

# OLS回帰
X = sm.add_constant(df['time'])
model = sm.OLS(df['price'], X)
results = model.fit()
df['y_hat'] = results.fittedvalues

# 残差と標準偏差
df['resid'] = df['price'] - df['y_hat']
sigma = df['resid'].std()
k = 2  # 標準偏差倍率

df['upper'] = df['y_hat'] + k * sigma
df['lower'] = df['y_hat'] - k * sigma

# プロット
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['price'], label='Price', color='gray', alpha=0.6)
plt.plot(df['time'], df['y_hat'], label='OLS Center', color='blue')
plt.plot(df['time'], df['upper'], label=f'+{k}σ Band', color='red', linestyle='--')
plt.plot(df['time'], df['lower'], label=f'-{k}σ Band', color='green', linestyle='--')
plt.fill_between(df['time'], df['upper'], df['lower'], color='orange', alpha=0.1)
plt.legend()
plt.title('OLS回帰直線＋標準偏差バンド（チャネル）')
plt.xlabel('Time')
plt.ylabel('Price')
plt.tight_layout()
plt.show() 