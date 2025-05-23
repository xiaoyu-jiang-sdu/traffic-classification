import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import random
import numpy as np


fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='nch_LLM')
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
classifier.to(device)

optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

X_train = torch.load(args.data_dir + 'X_train.pt', map_location="cpu")
y_train = torch.load(args.data_dir + 'y_train.pt', map_location="cpu")

X_valid = torch.load(args.data_dir + 'X_valid.pt', map_location="cpu")
y_valid = torch.load(args.data_dir + 'y_valid.pt', map_location="cpu")

train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers)

# 学习率调度器
if args.lradj == 'COS':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-8)
else:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                    steps_per_epoch=len(train_dataloader),
                                                    pct_start=args.pct_start,
                                                    epochs=args.train_epochs,
                                                    max_lr=args.learning_rate)
if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()

early_stopping_counter = 0
best_val_acc = 0.0
best_model_state = None

for epoch in range(args.train_epochs):
    classifier.train()
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.train_epochs}", unit="batch", mininterval=1)
    for X, y in pbar:
        X, y = X.to(device), y.long().to(device)

        optimizer.zero_grad()

        if args.use_amp:
            with torch.cuda.amp.autocast():
                output = classifier(X)
                loss = criterion(output, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = classifier(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # 更新学习率（如果使用OneCycleLR）
        if args.lradj != 'COS':
            scheduler.step()

        pbar.set_postfix(loss=loss.item())

    # 如果使用CosineAnnealingLR，每个epoch结束时更新
    if args.lradj == 'COS':
        scheduler.step()

    # 验证阶段
    classifier.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X, y in tqdm(valid_dataloader, desc="Validation", leave=False):
            data, target = X.to(device), y.long().to(device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    output = classifier(data)
                    loss = criterion(output, target)
            else:
                output = classifier(data)
                loss = criterion(output, target)

            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()

    val_acc = 100. * val_correct / val_total
    avg_val_loss = val_loss / len(valid_dataloader)

    print(f'\nEpoch {epoch + 1}/{args.train_epochs}:')
    print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

    # 早停和模型保存
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = classifier.state_dict().copy()
        early_stopping_counter = 0

        # 保存最佳模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'best_val_acc': best_val_acc,
            'args': args
        }, f'{args.checkpoints}/best_model.pth')

        print(f'New best validation accuracy: {best_val_acc:.2f}%')
    else:
        early_stopping_counter += 1

    # 早停检查
    if early_stopping_counter >= args.patience:
        print(f'\nEarly stopping triggered after {args.patience} epochs without improvement.')
        print(f'Best validation accuracy: {best_val_acc:.2f}%')
        break

    print(f'Early stopping counter: {early_stopping_counter}/{args.patience}')
    print('-' * 80)

# 训练结束后加载最佳模型
if best_model_state is not None:
    classifier.load_state_dict(best_model_state)
    print(f'\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%')
else:
    print('\nTraining completed without finding a better model.')

# 保存最终模型
torch.save({
    'model_state_dict': classifier.state_dict(),
    'args': args,
    'best_val_acc': best_val_acc
}, f'{args.checkpoints}/final_model.pth')

print(f'Models saved to {args.checkpoints}/')