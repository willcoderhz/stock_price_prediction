import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# 生成一些示例训练数据
X_train = np.array([[i] for i in range(1, 101)])  # 1 到 100 的天数
y_train = np.array([i * 2 for i in range(1, 101)])  # 假设股票价格是天数的 2 倍

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 将模型保存到 .pkl 文件中
with open('ml_models/stock_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("模型已成功保存到 stock_price_model.pkl 文件中")