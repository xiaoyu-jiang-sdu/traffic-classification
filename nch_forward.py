import argparse
import json

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_provider.dataloader import RawFlowDataset
from models import nchLLM

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
parser.add_argument('--checkpoints', type=str, default='./checkpoints/ckpt', help='location of models checkpoints')
parser.add_argument('--data_dir', type=str, default='./data/USTC-TFC2016/plain_forward/', help='location of train, valid and test data')

# forecasting task
parser.add_argument('--num_labels', type=int, default=20, help='labels num of specific task')
parser.add_argument('--label_names_json', type=str, default='./mapper/USTC-TFC.json', help='record label names json')
parser.add_argument('--categories_desc', type=str, default='./data/desc.txt', help='description of categories')

# models define
parser.add_argument('--d_model', type=int, default=768, help='dimension of models')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--n_layers', type=int, default=2, help='num of classifier layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='DEEPSEEK', help='LLM models')  # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='3584', help='LLM models dimension')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of models evaluation')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

# train
parser.add_argument('--from_ckpt', action="store_true", default=False, help="load from checkpoint")
parser.add_argument('--interrupt_it', type=int, default=0, help='training interrupt ckpt')
args = parser.parse_args()

with open(args.label_names_json, 'r') as jsonData:
    mapper = json.load(jsonData)

label_names = list(mapper.keys())
args.label_names = label_names

with open(args.categories_desc, 'r', encoding='utf-8') as file:
    content = file.read()
args.categories_desc = content

device = torch.device("cuda")

# 将模型、优化器、数据加载器等传递给 Accelerator 进行包装
model = nchLLM.Model(args).float()
model.to(device)

train_parameters = []
for param in model.parameters():
    if param.requires_grad:
        train_parameters.append(param)
dataset = RawFlowDataset('data/USTC-TFC2016/raw/test.csv', model.tokenizer)


def collate_fn(batch):
    flows, packet_nums, link_types, durations, labels = zip(*batch)

    # flows 是 list of [1, seq_len_i] -> squeeze 变成 [seq_len_i]
    flows = [f.squeeze(0) for f in flows]

    # 填充到 batch 内最长长度，batch_first 变成 [B, max_len]
    flows_padded = pad_sequence(flows, batch_first=True, padding_value=model.tokenizer.eos_token_id)

    packet_nums = torch.tensor(packet_nums, dtype=torch.float)
    link_types = torch.tensor(link_types, dtype=torch.float)
    durations = torch.tensor(durations, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    return flows_padded, packet_nums, link_types, durations, labels


dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
print("data load complete")
# validDataset = FlowLabelDataset('./data/valid.csv')
# val_dataloader = DataLoader(validDataset, batch_size=args.eval_batch_size, num_workers=args.num_workers)

if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()

print("forward start")
X = torch.tensor([]).to(device)
y = torch.tensor([]).to(device)

model.eval()
# 使用 tqdm 显示进度条
pbar = tqdm(dataloader, desc=f"LLM forward", unit="batch", mininterval=1)
for flow, packet_num, link_type, duration, label in pbar:
    flow = flow.to(device)
    packet_num = packet_num.to(device)
    link_type = link_type.to(device)
    duration = duration.to(device)
    label = label.to(device)

    if args.use_amp:
        with torch.amp.autocast(device_type="cuda"):
            x = model(flow, packet_num, link_type, duration)
            x = x.view(-1, model.d_llm)
            X = torch.cat([X, x], dim=0)
            y = torch.cat([y, label], dim=0)
    else:
        x = model(flow, packet_num, link_type, duration)
        x = x.view(-1, model.d_llm)
        X = torch.cat([X, x], dim=0)
        y = torch.cat([y, label], dim=0)

torch.save(X, args.data_dir + 'X_test.pt')
torch.save(y, args.data_dir + 'y_test.pt')
print("Forward complete!")
