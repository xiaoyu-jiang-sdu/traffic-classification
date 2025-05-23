import os
from utils.pcap_utils import split_cap, find_files


# 输入：pcap文件列表， 输出目录
# 将一个文件夹下的pcap文件按照会话拆分，按照名称区分文件夹
def split_pcap(pcap_files, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in pcap_files:
        dir = os.path.join(output_dir, os.path.basename(file).split('.')[0])
        split_cap(file, dir)


if __name__ == '__main__':
    pcap_files = find_files('E:/ChromeDownload/Tor/Tor')
    for pcap_file in pcap_files:
        split_pcap([pcap_file],'E:/ChromeDownload/Tor/split')
