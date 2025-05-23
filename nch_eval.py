import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import random
import numpy as np

parser = argparse.ArgumentParser(description='nch-LLM')

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--seed', type=int, default=2025, help='random seed')
# data loader
parser.add_argument('--checkpoints', type=str, default='./checkpoints/USTC-TFC2016-20cls-nch/', help='location of models checkpoints')
parser.add_argument('--data_dir', type=str, default='./data/USTC-TFC2016/plain_forward/', help='location of train, valid and test data')
# forecasting task
parser.add_argument('--num_labels', type=int, default=20, help='labels num of specific task')
parser.add_argument('--label_names_json', type=str, default='./mapper/USTC-TFC.json', help='record label names json')
# models define
parser.add_argument('--llm_dim', type=int, default='3584', help='LLM models dimension')
# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=16, help='batch size of models evaluation')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = nn.Linear(in_features=args.llm_dim, out_features=args.num_labels).float()
classifier.load_state_dict(torch.load(args.checkpoints + 'final_model.pth', weights_only=False)['model_state_dict'])

classifier.to(device)
classifier.eval()

X_test = torch.load(args.data_dir + 'X_valid.pt', map_location="cpu")
y_test = torch.load(args.data_dir + 'y_valid.pt', map_location="cpu")

test_dataset = TensorDataset(X_test, y_test)

dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

pbar = tqdm(dataloader, desc="eval", unit="batch", mininterval=10)

labels = torch.tensor([])
predicts = torch.tensor([])
with torch.no_grad():
    for X, y in tqdm(dataloader, desc="Testing", leave=False):
        data, target = X.to(device), y.long().to(device)

        if args.use_amp:
            with torch.cuda.amp.autocast():
                output = classifier(data)
        else:
            output = classifier(data)
        pred = torch.argmax(output, dim=1)
        labels = np.concatenate((labels, y.cpu().numpy()))
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