import os
import sys

from scapy.all import rdpcap
from tqdm import tqdm


# 查找给定目录下的所有pcap文件
def find_files(data_path, extension=".pcap"):
    pcap_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(extension):
                pcap_files.append(os.path.join(root, file))
    return pcap_files


# 将PCAPNG格式转换为PCAP格式
# pcapng文件路径，文件名，输出目录路径
# 需要wireshark的editcap，并添加env
def convert_pcapng_2_pcap(pcapng_path, pcapng_file_name, output_path):
    output_file = os.path.join(output_path, pcapng_file_name.replace(".pcapng", ".pcap"))
    cmd = "editcap -F pcap %s %s"
    command = cmd % (os.path.join(pcapng_path, pcapng_file_name), output_file)
    os.system(command)


# 按照packet/session切分pcap文件
# 使用SplitCap,一样需要添加env

def split_cap(pcap_file, output_dir, dataset_level='flow'):
    # 不存在输出目录则创建
    os.makedirs(output_dir, exist_ok=True)

    # 执行切分
    cmd = ''
    if dataset_level == 'flow':
        cmd = "SplitCap -r %s -s session -o " + output_dir
    elif dataset_level == 'packet':
        cmd = "SplitCap -r %s -s packets 1 -o " + output_dir
    command = cmd % pcap_file
    os.system(command)


# 使用tshark对packet进行过滤
def filter_pcap(pcap_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    pcap_files = find_files(pcap_dir)
    for pcap_file in pcap_files:
        filename = os.path.basename(pcap_file)
        output_file = os.path.join(output_dir, filename)
        # 根据协议类型过滤
        command = f'tshark -r "{pcap_file}" -Y "ip and not arp and not dhcp" -w "{output_file}"'
        os.system(command)

        print(f'clean file "{filename}"finish')

if __name__ == '__main__':
    output_dir = "E:/ChromeDownload/Tor/filtered"
    filter_pcap('E:/ChromeDownload/Tor/Tor',output_dir)
