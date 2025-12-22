import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# 数据
epochs = [1, 2, 3, 4, 5]
accuracy = [0.9203, 0.9711, 0.9721, 0.9747, 0.9761]
precision = [0.9249, 0.9712, 0.9723, 0.9749, 0.9772]
recall = [0.9182, 0.9699, 0.9709, 0.9735, 0.9766]
f1_score = [0.9189, 0.9701, 0.9711, 0.9737, 0.9758]

# 创建图像
plt.figure(figsize=(8, 6))

# 绘制折线图
plt.plot(epochs, accuracy, marker='o', label='Accuracy')
plt.plot(epochs, precision, marker='s', label='Precision')
plt.plot(epochs, recall, marker='^', label='Recall')
plt.plot(epochs, f1_score, marker='d', label='F1 Score')

# 设置标题和轴标签
plt.title('Test Metrics Over Epochs(USTC-TF2016 20cls)')
plt.xlabel('Epoch')
plt.ylabel('Score')

# 坐标轴美化
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.ylim(0.9, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()

# 显示图像
plt.show()
