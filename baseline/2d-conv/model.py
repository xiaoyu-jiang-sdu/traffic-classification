import torch.nn as nn
import torch.nn.functional as F


class CNN2DBaseLine(nn.Module):
    def __init__(self, num_classes=20):
        super(CNN2DBaseLine, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # [B,32,28,28]
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # [B,32,28,28]
        self.pool = nn.MaxPool2d(2, 2)  # [B,32,14,14]

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # [B,64,14,14]
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # [B,64,14,14]
        # pool to [B,64,7,7]

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 28->14

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)  # 14->7

        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x