from utils.pcap_utils import filter_pcap
from utils.pcap_utils import find_files


def clean_data(pcap_dir):
    pcap_files = find_files(pcap_dir)
    for file in pcap_files:
        # 过滤不符合要求的pcap
        filter_pcap(file)


if __name__ == '__main__':
    clean_data('E:/ChromeDownload/USTC-TFC2016-master/dataset')