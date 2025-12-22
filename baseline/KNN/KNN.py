from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from torch.utils.data import DataLoader
from baseline.dataloader import FlowDataset
import numpy as np

train_dataset = FlowDataset('../../data/USTC-TFC2016/raw/train.csv')

dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))  # 一次性加载所有样本
X_train_list, y_train_list = [], []

for X_batch, y_batch in dataloader:
    X_flattened = X_batch.view(X_batch.size(0), -1)  # shape: [N, 5*100] = [N, 500]
    X_train_list.append(X_flattened.numpy())
    y_train_list.append(y_batch.numpy())

# 拼接所有数据
X = np.concatenate(X_train_list, axis=0)
y = np.concatenate(y_train_list, axis=0)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

test_dataset = FlowDataset('../../data/USTC-TFC2016/raw/test.csv')
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))  # 全部加载

X_test_list, y_test_list = [], []
for X_batch, y_batch in test_loader:
    X_batch = X_batch.view(X_batch.size(0), -1)
    X_test_list.append(X_batch.numpy())
    y_test_list.append(y_batch.numpy())

X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

# 预测与评估
y_pred = knn.predict(X_test)

# 计算准确率 (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 计算精确率 (Precision)
precision = precision_score(y_test, y_pred, average='macro')  # 使用 'macro' 平均
print(f"Precision: {precision:.4f}")

# 计算召回率 (Recall)
recall = recall_score(y_test, y_pred, average='macro')  # 使用 'macro' 平均
print(f"Recall: {recall:.4f}")

# 计算F1分数 (F1 Score)
f1 = f1_score(y_test, y_pred, average='macro')  # 使用 'macro' 平均
print(f"F1 Score: {f1:.4f}")

confusion_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix:{confusion_matrix}")

report = classification_report(y_test, y_pred)
print(f"Classification report: {report}")
