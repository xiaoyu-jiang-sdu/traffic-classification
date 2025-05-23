from torch.utils.data import Dataset
import pandas as pd
import torch


class FlowLabelDataset(Dataset):
    def __init__(self, tsv_file):
        self.data = pd.read_csv(tsv_file)
        self.data["flow"] = self.data["flow"].apply(lambda x: x.split(";")[:-1] if isinstance(x, str) else [])
        self.data["count"] = self.data["count"]
        self.data["link_type"] = self.data["link_type"]
        self.data["duration"] = self.data["duration"]
        self.data["label"] = self.data["label"]

    def __len__(self):
        return len(self.data['flow'])

    def __getitem__(self, idx):
        # 返回每个流量数据
        flow = self.data.iloc[idx]["flow"]
        label = self.data.iloc[idx]["label"]
        packet_num = self.data.iloc[idx]['count']
        link_type = self.data.iloc[idx]['link_type']
        duration = self.data.iloc[idx]['duration']
        headers, payloads = flow_2_header_payload_tensor(flow)
        return headers, payloads, packet_num, link_type, duration, label


def flow_2_header_payload_tensor(flow):
    headers = []
    payloads = []
    for packet in flow:
        packet_bytes = packet.split(' ')
        headers.append([int(header, 16) for header in packet_bytes[:80]])
        payloads.append([int(payload, 16) for payload in packet_bytes[80:]])
    headers = torch.tensor(headers)
    payloads = torch.tensor(payloads)
    return headers, payloads


class RawFlowDataset(Dataset):
    def __init__(self, tsv_file, tokenizer):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(tsv_file)
        self.data["flow"] = self.data["flow"]
        self.data["count"] = self.data["count"]
        self.data["link_type"] = self.data["link_type"]
        self.data["duration"] = self.data["duration"]
        self.data["label"] = self.data["label"]

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        # 返回每个流量数据
        flow = self.data.iloc[idx]["flow"]
        label = self.data.iloc[idx]["label"]
        packet_num = self.data.iloc[idx]['count']
        link_type = self.data.iloc[idx]['link_type']
        duration = self.data.iloc[idx]['duration']
        flow = self.tokenizer(flow, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        return flow, packet_num, link_type, duration, label
