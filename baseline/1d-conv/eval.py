import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CNN1DBaseLine
from baseline.dataloader import FlowDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1DBaseLine(num_classes=20).to(device)
model.load_state_dict(torch.load('ckpt/best_model-ustc-tfc20cls.pth'))

criterion = nn.CrossEntropyLoss()

test_dataset = FlowDataset('../../data/USTC-TFC2016/raw/test.csv')
test_dataloader = DataLoader(test_dataset, batch_size=32)

model.eval()

labels = np.array([])
predicts = np.array([])

pbar = tqdm(test_dataloader, desc=f"eval", unit="batch", mininterval=1)
for flow_tensor, label in pbar:
    with torch.no_grad():
        flow_tensor = flow_tensor.unsqueeze(1).to(device)
        label = label.to(device)
        output = model(flow_tensor)
        loss = criterion(output, label)
        pred = torch.argmax(output, dim=1)
        pbar.set_postfix(loss=loss.item())
    labels = np.concatenate((labels, label.cpu().numpy()))
    predicts = np.concatenate((predicts, pred.cpu().numpy()))

# 计算准确率 (Accuracy)
accuracy = accuracy_score(labels, predicts)
print(f"Accuracy: {accuracy:.4f}")

# 计算精确率 (Precision)
precision = precision_score(labels, predicts, average='macro')  # 使用 'macro' 平均
print(f"Precision: {precision:.4f}")

# 计算召回率 (Recall)
recall = recall_score(labels, predicts, average='macro')  # 使用 'macro' 平均
print(f"Recall: {recall:.4f}")

# 计算F1分数 (F1 Score)
f1 = f1_score(labels, predicts, average='macro')  # 使用 'macro' 平均
print(f"F1 Score: {f1:.4f}")

confusion_matrix = confusion_matrix(labels, predicts)
print(f"Confusion matrix:{confusion_matrix}")

report = classification_report(labels, predicts)
print(f"Classification report: {report}")
