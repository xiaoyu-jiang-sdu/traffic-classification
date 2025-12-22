import argparse
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data_provider.dataloader import FlowLabelDataset
from models import TCLLM

import random
import numpy as np

parser = argparse.ArgumentParser(description='TC-LLM')

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--seed', type=int, default=2025, help='random seed')

# data loader
parser.add_argument('--checkpoints', type=str, default='./checkpoints/USTC-TFC2016-20cls/', help='location of models checkpoints')
parser.add_argument('--data_dir', type=str, default='./data/USTC-TFC2016/20cls/', help='location of train, valid and test data')

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
parser.add_argument('--llm_model', type=str, default='DEEPSEEK', help='LLM models')  # LLAMA, GPT2, BERT, DEEPSEEK
parser.add_argument('--llm_dim', type=int, default='3584', help='LLM models dimension')
parser.add_argument('--no_reprogram', action="store_true", default=False, help='whether using reprogramming layer')
parser.add_argument('--content', type=str, default='')
# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of models evaluation')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
# train
parser.add_argument('--from_ckpt', action="store_true", default=False, help="load from checkpoint")
parser.add_argument('--interrupt_it', type=int, default=0, help='training interrupt ckpt')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(args.label_names_json, 'r') as jsonData:
    mapper = json.load(jsonData)

label_names = ' '.join(list(mapper.keys()))
args.label_names = label_names

# 将模型、优化器、数据加载器等传递给 Accelerator 进行包装
model = TCLLM.Model(args).float()
model.to(device)

trained_parameters = []
for p in model.parameters():
    if p.requires_grad is True:
        trained_parameters.append(p)

optimizer = optim.Adam(trained_parameters, lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

dataset = FlowLabelDataset(args.data_dir +  'train.csv')
if args.from_ckpt:
    dataset = Subset(dataset, range(args.interrupt_it, len(dataset)))
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

# validDataset = FlowLabelDataset('data/20cls/valid.csv')
# val_dataloader = DataLoader(validDataset, batch_size=args.eval_batch_size, num_workers=args.num_workers)

# 学习率调度器
if args.lradj == 'COS':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-8)
else:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                    steps_per_epoch=len(dataloader),
                                                    pct_start=args.pct_start,
                                                    epochs=args.train_epochs,
                                                    max_lr=args.learning_rate)

if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()

if args.from_ckpt:
    model.load_model_states(args.checkpoints)

best_loss = float('inf')
early_stopping_counter = 0

for epoch in range(args.train_epochs):
    model.train()
    total_loss = 0
    item = 0
    state_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.train_epochs}", unit="batch", mininterval=1)
    for headers, payloads, packet_num, link_type, duration, label in pbar:
        optimizer.zero_grad()

        if args.use_amp:
            with torch.amp.autocast(device_type="cuda"):
                header_tensor = headers.float().to(device)
                payload_tensor = payloads.float().to(device)
                label = label.to(device)
                x = model(header_tensor, payload_tensor, packet_num, link_type, duration).to(device)
                loss = criterion(x, label)
        else:
            header_tensor = headers.float().to(device)
            payload_tensor = payloads.float().to(device)
            label = label.to(device)
            x = model(header_tensor, payload_tensor, packet_num, link_type, duration).to(device)
            loss = criterion(x, label)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        state_loss += loss.item()
        item += 1

        pbar.set_postfix(loss=loss.item())

        if item % 1000 == 0:
            state_loss /= 1000
            model.save_model_states(args.checkpoints)
            print(f"save state on train it {item} with loss {state_loss}")
            state_loss = 0

    avg_loss = total_loss / len(dataloader)
    model.save_model_states(args.checkpoints)
    print(f"Epoch [{epoch + 1}/{args.train_epochs}], Loss: {avg_loss:.4f}")

    # # 验证阶段
    # model.eval()
    # total_val_loss = 0
    # correct = 0
    # total = 0
    #
    # with torch.no_grad():
    #     for headers, payloads, packet_num, link_type, duration, label in pbar:
    #         header_tensor = headers.float().to(device)
    #         payload_tensor = payloads.float().to(device)
    #
    #         outputs = model(headers, payloads, packet_num, link_type, duration).to(device)
    #         predicted = torch.argmax(outputs, dim=1)
    #
    #         correct += (predicted == label).sum().item()
    #         total += label.size(0)
    #
    #         acc = correct / total
    #         pbar.set_postfix(acc=f"{acc:.4f}")
    # avg_val_loss = total_val_loss / len(val_dataloader)
    # val_accuracy = correct / total
    # print(
    #     f"Epoch [{epoch + 1}/{args.train_epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    #
    # # 保存最佳模型
    # if avg_val_loss < best_loss:
    #     best_loss = avg_val_loss
    #     model.save_model_states(args.checkpoints + 'best/')
    #     print(f"Model saved with best val loss {best_loss:.4f} at epoch {epoch + 1}")
    #     early_stopping_counter = 0
    # else:
    #     early_stopping_counter += 1
    #     if early_stopping_counter >= args.patience:
    #         print("Early stopping triggered")
    #         break
    #
    # # 更新学习率
    # if args.lradj == 'COS':
    #     scheduler.step()
    # else:
    #     scheduler.step()

print("Training complete!")