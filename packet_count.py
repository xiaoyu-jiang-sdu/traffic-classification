import os
from scapy.all import rdpcap

def count_total_packets_by_folder(folder_path, packet_count):
    total_packets = 0
    for root, dirs, files in os.walk(folder_path):
        # 只处理当前文件夹下的 pcap 文件
        pcap_files = sorted([f for f in files if f.endswith(".pcap")])
        if not pcap_files:
            continue

        # 除最后一个文件，其他文件固定 packet_count
        num_files = len(pcap_files)
        total_packets += (num_files - 1) * packet_count

        # 最后一个文件实际包数量
        last_file_path = os.path.join(root, pcap_files[-1])
        try:
            last_packets = len(rdpcap(last_file_path))
            total_packets += last_packets
            print(f"{last_file_path}: {last_packets} packets")
        except Exception as e:
            print(f"Failed to read {last_file_path}: {e}")

    return total_packets
if __name__ == "__main__":
    folder_path = r"E:\ChromeDownload\Tor\split\voip"
    packet_count = 100  # 每个文件的固定包数
    total = count_total_packets_by_folder(folder_path, packet_count)
    print(f"Total packets (considering fixed split): {total}")