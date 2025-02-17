import numpy as np
from sklearn.covariance import MinCovDet

# 生成一维数据（模拟情况）
data = np.array([10, 12, 10, 14, 100, 15, 11, 13, 12, 11]).reshape(-1, 1)

# 初始化Deterministic Minimum Covariance Determinant（MinCovDet）
mcd = MinCovDet()

# 拟合数据并计算MCD估计的中心和协方差
mcd.fit(data)

# 输出结果
print(f"Robust center: {mcd.location_}")
print(f"Robust covariance (in 1D, it should be variance): {mcd.covariance_}")
