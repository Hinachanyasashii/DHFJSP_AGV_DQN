# import matplotlib.pyplot as plt
#
# import pandas as pd
#
# # 读取 Excel 文件中的数据
# df = pd.read_excel("Result/10J2F2A.xlsx")
# df['TR'] = df['TR'].abs()
#
# # 将数据从 DataFrame 转换为列表
# TR = df['TR'].tolist()  # 假设数据在名为 'TR' 的列中
#
# # print(TR)  # 输出列表内容以确认读取成功
# Q1 = df['TR'].quantile(0.25)
# Q3 = df['TR'].quantile(0.75)
# IQR = Q3 - Q1
# print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
#
#
# # 绘制箱型图
# plt.figure(figsize=(8, 6))
# plt.boxplot(TR, vert=True, patch_artist=True)  # vert=True 代表垂直箱型图
# plt.title("Box Plot of TR Data")
# plt.xlabel("TR")
# plt.ylabel("Values")
# plt.show()

import numpy as np

# 生成一个10x10的矩阵，每个元素为0、2、4、6、8、10、12中的一个
matrix = np.random.choice([2, 4, 6, 8, 10, 12], size=(10, 10))

# # 使矩阵对称
# symmetric_matrix = np.tril(matrix) + np.tril(matrix, -1).T

# 将对角线元素设置为0
np.fill_diagonal(matrix, 0)

# 打印对称矩阵
print(matrix)



