import pandas as pd
import torch
from torch.utils.data import Dataset


# 不考虑分号
def hexstr_to_bytes(hex_str):
    hex_str = hex_str.replace(";", " ").strip()
    tokens = hex_str.split()
    bytes_arr = []
    for token in tokens:
        try:
            b = int(token, 16)
            bytes_arr.append(b)
        except:
            continue
    return bytes_arr


class FlowDataset(Dataset):
    def __init__(self, tsv_file, max_len=100):
        data = pd.read_csv(tsv_file)
        self.flow = data['flow']
        self.label = data['label']
        self.max_len = max_len

    def __len__(self):
        return len(self.flow)

    def __getitem__(self, idx):
        flow_bytes = self.flow.iloc[idx]
        label = self.label.iloc[idx]

        flow_bytes = hexstr_to_bytes(flow_bytes)

        # 截断或补零（0~255）
        if len(flow_bytes) > self.max_len:
            flow_bytes = flow_bytes[:self.max_len]
        else:
            flow_bytes = flow_bytes + [0] * (self.max_len - len(flow_bytes))

        flow_tensor = torch.tensor(flow_bytes, dtype=torch.float32)

        return flow_tensor, label


class PacketDataset(Dataset):
    def __init__(self, tsv_file, max_packet_len=100, num_packets=5):
        data = pd.read_csv(tsv_file)
        self.flow = data['flow']
        self.label = data['label']
        self.max_packet_len = max_packet_len
        self.num_packets = num_packets

    def __len__(self):
        return len(self.flow)

    def __getitem__(self, idx):
        flow = self.flow.iloc[idx]
        label = self.label.iloc[idx]

        # 按 ; 拆分 packet
        packets = flow.split(";")

        # 取固定数量的 packet
        if len(packets) >= self.num_packets:
            packets = packets[:self.num_packets]
        else:
            # 如果不够，补充空 packet
            packets += [""] * (self.num_packets - len(packets))

        packet_bytes_list = []
        for pkt in packets:
            bytes_arr = hexstr_to_bytes(pkt)

            # 截断或补零
            if len(bytes_arr) > self.max_packet_len:
                bytes_arr = bytes_arr[:self.max_packet_len]
            else:
                bytes_arr += [0] * (self.max_packet_len - len(bytes_arr))

            packet_bytes_list.append(bytes_arr)

        flow_tensor = torch.tensor(packet_bytes_list, dtype=torch.float32)

        return flow_tensor, label  # 用于 LSTM 输入



if __name__ == '__main__':
    dataset = PacketDataset('../data/USTC-TFC2016/raw/test.csv')
    print(dataset[0])