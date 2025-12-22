import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LSTMBaseLine
from baseline.dataloader import PacketDataset

batch_size = 32
lr = 1e-3
epochs = 30
early_stop_patience = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMBaseLine(100, num_classes=20).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_dataset = PacketDataset('../../data/USTC-TFC2016/raw/train.csv')
val_dataset = PacketDataset('../../data/USTC-TFC2016/raw/valid.csv')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

best_loss = float('inf')
early_stopping_counter = 0
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", mininterval=1)
    for flow_tensor, label in pbar:
        flow_tensor = flow_tensor.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(flow_tensor)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

        pbar.set_postfix(loss=loss.item())

    train_acc = correct / total
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}")

    # ---------------- 验证 ----------------
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for flow_tensor, label in val_dataloader:
            flow_tensor = flow_tensor.to(device)
            label = label.to(device)

            output = model(flow_tensor)
            loss = criterion(output, label)
            val_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

    avg_val_loss = val_loss / len(val_dataloader)
    val_acc = correct / total
    print(f"Epoch {epoch + 1}: Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")

    # ---------------- Early Stopping & Save ----------------
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), "ckpt/best_model-ustc-tfc20cls.pth")
        print(f"Saved new best model at epoch {epoch + 1}")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break
