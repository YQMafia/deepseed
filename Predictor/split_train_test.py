import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox, skew
import numpy as np

# 读取数据文件（这里假设是CSV文件，且第一行为列名，根据实际情况调整）
data = pd.read_csv('data/random_seq_v2.csv')

# 提取特征数据（假设除最后一列是目标列外，前面都是特征列，可根据实际情况调整）
features = data.iloc[:, 1:-2].values + 0.000001
train_data_boxcox, test_data_boxcox = train_test_split(data, test_size=0.05, random_state=42)
train_data_boxcox.to_csv('data/predictor_train_.csv', index=False)
test_data_boxcox.to_csv('data/predictor_test_.csv', index=False)

# # 1. 标准归一化（Z - score标准化）
# scaler_standard = StandardScaler()
# normalized_features_standard = scaler_standard.fit_transform(features)
# data_standard = data.copy()
# data_standard.iloc[:, 1:-2] = normalized_features_standard 
# train_data_standard, test_data_standard = train_test_split(data_standard, test_size=0.05, random_state=42)
# train_data_standard.to_csv('data/predictor_train_SS.csv', index=False)
# test_data_standard.to_csv('data/predictor_test_SS.csv', index=False)

# # 2. 鲁棒归一化（RobustScaler）
# scaler_robust = RobustScaler()
# normalized_features_robust = scaler_robust.fit_transform(features)
# data_robust = data.copy()
# data_robust.iloc[:, 1:-2] = normalized_features_robust 
# train_data_robust, test_data_robust = train_test_split(data_robust, test_size=0.05, random_state=42)
# train_data_robust.to_csv('data/predictor_train_RS.csv', index=False)
# test_data_robust.to_csv('data/predictor_test_RS.csv', index=False)

# # 3. Box - Cox归一化（注意Box - Cox要求数据为正，这里先做简单处理使数据都大于0）
# # 找到特征数据中的最小值
# min_value = np.min(features)
# shifted_features = features - min_value + 1  # 让所有数据大于0
# transformed_features = []
# for col in shifted_features.T:
#     transformed_col, _ = boxcox(col)
#     transformed_features.append(transformed_col)
# transformed_features = np.array(transformed_features).T
# data_boxcox = data.copy()
# data_boxcox.iloc[:, 1:-2] = transformed_features
# train_data_boxcox, test_data_boxcox = train_test_split(data_boxcox, test_size=0.05, random_state=42)
# train_data_boxcox.to_csv('data/predictor_train_BC.csv', index=False)
# test_data_boxcox.to_csv('data/predictor_test_BC.csv', index=False)



# # 3. Box - Cox归一化（注意Box - Cox要求数据为正，这里先做简单处理使数据都大于0）
# # 找到特征数据中的最小值
# original_skew = skew(features.flatten())
# if original_skew > 0:
#     lmbda = 0.5  # 正偏态时选择一个小于1的值，这里示例选0.5
# elif original_skew < 0:
#     lmbda = 1.5  # 负偏态时选择一个大于1的值，这里示例选1.5
# else:
#     lmbda = 0  # 如果偏度接近0，可选择0（等价于自然对数变换）
# transformed_features = []
# for col in features.T:
#     transformed_col = boxcox(col, lmbda=lmbda)
#     transformed_features.append(transformed_col)

# transformed_features = np.array(transformed_features).T
# data_boxcox = data.copy()
# data_boxcox.iloc[:, 1:-2] = transformed_features
# train_data_boxcox, test_data_boxcox = train_test_split(data_boxcox, test_size=0.05, random_state=42)
# train_data_boxcox.to_csv('data/predictor_train_BCL.csv', index=False)
# test_data_boxcox.to_csv('data/predictor_test_BCL.csv', index=False)
