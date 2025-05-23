import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Classifier(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, output_feature, dropout=0.1):
        super(Classifier, self).__init__()
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ll = nn.Linear(embed_dim, output_feature)

    def forward(self, x):
        for layer in self.transformer:
            x = layer(x)  # B * L * D
        feature = x[:, -1, :] # B * 1 * D
        x = feature.squeeze(1)
        logits = self.ll(x)
        return logits  # B * D
