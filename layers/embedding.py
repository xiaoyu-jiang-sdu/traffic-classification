import math

import torch.nn as nn
import torch


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class ConvEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(ConvEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular', bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        B, P, T, _ = x.shape
        x = x.float()
        x = x.view(B * P, 1, T)
        x = self.tokenConv(x)  # (B*P, d_model, T)
        x = x.transpose(1, 2)  # (B*P, T, d_model)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.view(B, P, T, -1)
        return x


class StrideEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1):
        super(StrideEmbedding, self).__init__()
        self.stride_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.stride_embedding(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class PacketEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, scale_factor=10):
        super(PacketEmbedding, self).__init__()
        self.packet_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.uniform_(self.packet_embedding.weight, a=-scale_factor, b=scale_factor)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.packet_embedding(x)
        x = self.activation(x)
        return x


class Embedding(nn.Module):
    def __init__(self, d_model=768, max_len=5000, max_packet_num=5,
                 header_vocab_size=256, dropout=0.1):
        super(Embedding, self).__init__()
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.payload_embedding = ConvEmbedding(d_model=d_model, c_in=1, dropout=dropout)
        self.header_embedding = StrideEmbedding(d_model=d_model, vocab_size=header_vocab_size, dropout=dropout)
        self.packet_embedding = PacketEmbedding(d_model=d_model, vocab_size=max_packet_num)

    def forward(self, headers, payloads):
        assert headers.device == payloads.device
        # headers:B * P * T -> B * P * T * N
        # payloads:B * P * T -> B * P * 1 * T -> B * P * T * N
        B, P, T = headers.shape[:3]
        _, _, L = payloads.shape[:3]
        packet_idx = torch.arange(P, device=headers.device).unsqueeze(0).expand(B, -1)  # (B, P)

        headers = headers.int()
        headers = self.header_embedding(headers)
        pos_emb = self.position_embedding(headers.view(-1, headers.size(2)))
        headers = headers + pos_emb
        payloads = self.payload_embedding(payloads.unsqueeze(-1))
        pos_emb = self.position_embedding(payloads.view(-1, payloads.size(2)))
        payloads = payloads + pos_emb
        packet_emb = self.packet_embedding(packet_idx)  # (B, P, d_model)
        packet_emb = packet_emb.unsqueeze(2).expand(-1, -1, T, -1)

        headers = headers + packet_emb
        packet_emb = self.packet_embedding(packet_idx)  # (B, P, d_model)
        packet_emb = packet_emb.unsqueeze(2).expand(-1, -1, L, -1)
        payloads = payloads + packet_emb
        return headers, payloads

