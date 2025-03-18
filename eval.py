import argparse
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_provider.dataloader import FlowLabelDataset
from models import TCLLM

import random
import numpy as np
from collections import OrderedDict

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

parser = argparse.ArgumentParser(description='TC-LLM')
fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--seed', type=int, default=2025, help='random seed')

# data loader
parser.add_argument('--merge_ckpt', type=str, default='./merge_ckpt/', help='location of models checkpoints')

# forecasting task
parser.add_argument('--num_labels', type=int, default=20, help='labels num of specific task')
parser.add_argument('--label_names_json', type=str, default='./mapper/USTC-TFC.json', help='record label names json')

# models define
parser.add_argument('--d_model', type=int, default=768, help='dimension of models')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM models')  # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM models dimension')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of models evaluation')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()

with open(args.label_names_json, 'r') as jsonData:
    mapper = json.load(jsonData)

label_names = ' '.join(list(mapper.keys()))
args.label_names = label_names


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
model = TCLLM.Model(args).float()

dataset = FlowLabelDataset('./data/test.csv')
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
index_file_path = f"{args.merge_ckpt}/pytorch_model.bin.index.json"

with open(index_file_path, "r") as f:
    index_data = json.load(f)

shard_files = index_data["weight_map"].values()
state_dict = OrderedDict()
for shard_file in set(shard_files):
    shard_path = f"{args.merge_ckpt}/{shard_file}"
    shard_state_dict = torch.load(shard_path, map_location="cpu")
    state_dict.update(shard_state_dict)

model.load_state_dict(state_dict)

labels = np.array([])
predicts = np.array([])

model.to(device )
model.eval()
pbar = tqdm(dataloader, desc="eval", unit="batch", mininterval=10)
for headers, payloads, packet_num, link_type, duration, label in pbar:
    with torch.no_grad():
        headers = headers.to(device).float()
        payloads = payloads.to(device).float()
        x = model(headers, payloads, packet_num, link_type, duration).to(device)
        loss = criterion(x, label.to(device))
        pred = torch.argmax(x, dim=1)
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


#Accuracy: 0.7626
#Precision: 0.7927
#Recall: 0.7628
#F1 Score: 0.7502
#