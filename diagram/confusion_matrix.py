import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

preds = np.load('../results/Tor-8clspredicts.npy')
labels = np.load('../results/Tor-8clslabels.npy')

with open("../mapper/Tor.json") as f:
    class_names = list(json.load(f).keys())

# 计算混淆矩阵
cm = confusion_matrix(labels, preds)

# 归一化选
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 创建图像
fig, ax = plt.subplots(figsize=(8, 6))  # 适当调大图像尺寸
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)

# 绘制归一化混淆矩阵
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f")

# 旋转 x 轴标签防止重叠
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.title("Confusion Matrix(Tor 8cls)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
