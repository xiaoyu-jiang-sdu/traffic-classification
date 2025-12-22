import torch.nn as nn
import torch.nn.functional as F


class LSTMBaseLine(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=20):
        super(LSTMBaseLine, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.5, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [B, T, F]
        out, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        logits = self.fc(h)
        return logits