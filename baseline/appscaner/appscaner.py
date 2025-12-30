import json
import os
import numpy as np
from appscanner.preprocessor import Preprocessor
from appscanner.appscanner import AppScanner
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, accuracy_score

# ------------------------------
# 1. 数据集和标签映射
# ------------------------------
base_dir = r"E:/ChromeDownload/Tor/flows_sampled"
label_map_path = r"../../mapper/Tor.json"

with open(label_map_path, "r", encoding="utf-8") as f:
    label_map = json.load(f)

pcap_paths = []
labels_str = []

for label in os.listdir(base_dir):
    label_dir = os.path.join(base_dir, label)
    if not os.path.isdir(label_dir):
        continue
    for file in os.listdir(label_dir):
        if file.endswith(".pcap"):
            pcap_paths.append(os.path.join(label_dir, file))
            labels_str.append(label)  # 先保留字符串标签

# ------------------------------
# 2. AppScanner 预处理
# ------------------------------
preprocessor = Preprocessor()
X, y_str = preprocessor.process(pcap_paths, labels_str)

# ------------------------------
# 3. 标签转换为整数
# ------------------------------
y = np.array([label_map[l] for l in y_str], dtype=int)

# ------------------------------
# 4. 划分训练集和测试集
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2025, stratify=y
)

# ------------------------------
# 5. 训练 AppScanner
# ------------------------------
scanner = AppScanner(threshold=0.0)  # 阈值设为0，避免Unknown
scanner.fit(X_train, y_train)

# ------------------------------
# 6. 预测
# ------------------------------
y_pred = scanner.predict(X_test)

# ------------------------------
# 7. 计算并输出指标
# ------------------------------
all_labels = list(range(8))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

cm = confusion_matrix(y_test, y_pred, labels=all_labels)
report = classification_report(y_test, y_pred, labels=all_labels, zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion matrix:\n{cm}")
print(f"Classification report:\n{report}")