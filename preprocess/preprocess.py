import binascii
import os.path
import json
import csv

import numpy as np
from scapy.all import rdpcap, PcapReader
import pandas as pd

from utils.pcap_utils import find_files


# 将packet的网络层部分转换成str
# 只对IP的地址进行可选掩码
# 返回完整的header和payload
def packet_2_str(packet, remove_ip=False, padding=True):
    if packet.haslayer('IP'):
        ip = packet['IP']
    else:
        ip = packet['IPv6']
    if remove_ip:
        pad_ip_addr = "0.0.0.0"
        ip.src, ip.dst = pad_ip_addr, pad_ip_addr
    header = (binascii.hexlify(bytes(ip))).decode()
    try:
        payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
        header = header.replace(payload, '')
    except:
        payload = ''
    if padding:
        header = header[:160] if len(header) > 160 else header + '0' * (160 - len(header))
        payload = payload[:480] if len(payload) > 480 else payload + '0' * (480 - len(payload))
    return header, payload


# 将flow字符串切分
def string_2_hex_array(flow_string, step=2, sliding_window=0):
    assert step > sliding_window
    if sliding_window == 0:
        return np.array([flow_string[i:i + step] for i in range(0, len(flow_string), step)])
    else:
        return np.array([flow_string[i:i + step] for i in range(0, len(flow_string) - step + 1, sliding_window)])


# 读取切分后的pcap文件，并进行过滤，分割
# 输入：pcap文件路径
def read_flow(pcap_files, label, max_packet_num=5, remove_ip=False):
    data = []
    for pcap_file in pcap_files:
        with PcapReader(pcap_file) as pcap_reader:
            link_type = pcap_reader.linktype
        packets = rdpcap(pcap_file)
        flow_bytes = ''
        flow_padding = ' '.join('00' for i in range(80)) + ' ' + ' '.join('0000' for i in range(239)) + ';'
        for packet in packets[:max_packet_num]:
            header, payload = packet_2_str(packet, remove_ip)
            flow_bytes += ' '.join(string_2_hex_array(header)) + ' '
            flow_bytes += ' '.join(string_2_hex_array(payload, 4, 2))
            flow_bytes += ';'
        if max_packet_num > len(packets):
            flow_bytes += flow_padding * (max_packet_num - (len(packets)))
        duration = packets[-1].time - packets[0].time
        data.append([label, len(packets), link_type, duration, flow_bytes])
    return data


def read_raw_flow(pcap_files, label, max_packet_num=5, remove_ip=True):
    data = []
    for pcap_file in pcap_files:
        with PcapReader(pcap_file) as pcap_reader:
            link_type = pcap_reader.linktype
        packets = rdpcap(pcap_file)
        flow_bytes = ''
        for packet in packets[:max_packet_num]:
            header, payload = packet_2_str(packet, remove_ip=remove_ip, padding=False)
            flow_bytes += ' '.join(string_2_hex_array(header)) + ' '
            flow_bytes += ' '.join(string_2_hex_array(payload))
            flow_bytes += ';'
        duration = packets[-1].time - packets[0].time
        data.append([label, len(packets), link_type, duration, flow_bytes])
    return data


def preprocess_USTC_TFC():
    path = "E:/ChromeDownload/USTC-TFC2016-master/flows_sampled"
    with open('../mapper/USTC-TFC.json', 'r') as jsonData:
        mapper = json.load(jsonData)

    data = []
    csv_path = '../data/USTC-TFC2016/20cls/20cls.csv'
    if not os.path.exists(csv_path):
        df = pd.DataFrame(data, columns=['label', 'count', 'link_type', 'duration', 'flow'])
        df.to_csv(csv_path, index=False)

    dirs = os.listdir(path)
    for dirname in dirs:
        label = mapper[dirname]
        pcap_files = find_files(os.path.join(path, dirname))
        data = read_flow(pcap_files, label)
        with open(csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)


def preprocess_Tor():
    path = "E:/ChromeDownload/Tor/flows_sampled"
    with open('../mapper/Tor.json', 'r') as jsonData:
        mapper = json.load(jsonData)

    data = []
    csv_path = '../data/Tor/raw/raw.csv'
    if not os.path.exists(csv_path):
        df = pd.DataFrame(data, columns=['label', 'count', 'link_type', 'duration', 'flow'])
        df.to_csv(csv_path, index=False)

    dirs = os.listdir(path)
    for dirname in dirs:
        label = mapper[dirname]
        pcap_files = find_files(os.path.join(path, dirname))
        data = read_raw_flow(pcap_files, label, remove_ip=True)
        with open(csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)


if __name__ == '__main__':
    preprocess_USTC_TFC()
    # preprocess_Tor()
