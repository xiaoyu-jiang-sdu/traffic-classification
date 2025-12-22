import os
import subprocess

from scapy.utils import rdpcap

from utils.pcap_utils import split_cap, split_cap_time, find_files


# 输入：pcap文件列表， 输出目录
# 将一个文件夹下的pcap文件按照会话拆分，按照名称区分文件夹
# 默认根据session划分
def split_pcap(pcap_files, output_dir, session=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in pcap_files:
        dir = os.path.join(output_dir, os.path.basename(file).split('.')[0])
        if session:
            split_cap(file, dir)
        else: # 反之根据 1s间隔划分
            split_cap_time(file, dir)


def delete_small_pcaps(root_dir, min_packets=5):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".pcap"):
                full_path = os.path.join(dirpath, filename)
                try:
                    # 使用 tshark 统计包数量
                    cmd = ["tshark", "-r", full_path, "-T", "fields", "-e", "frame.number"]
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                    packet_count = result.stdout.count("\n")

                    if packet_count < min_packets:
                        print(f"Deleting {full_path} ({packet_count} packets)")
                        os.remove(full_path)
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")


if __name__ == '__main__':
    # dir = 'E:/ChromeDownload/Tor/filtered'
    # pcap_files = find_files(dir)
    flow_packet_num_dict = {
        "BROWSING": 60,
        "CHAT": 100,
        "AUDIO": 100,
        "VIDEO": 200,
        "FILETRANSFER": 300,
        "MAIL": 300,
        "P2P": 150,
        "VOIP": 100,
    }
    # for pcap_file in pcap_files:
    #     split_pcap([pcap_file],'E:/ChromeDownload/Tor/split', True)
    # delete_small_pcaps('E:/ChromeDownload/Tor/split')

    # for label, packets_num in flow_packet_num_dict.items():
    #     class_dir = os.path.join('E:/ChromeDownload/Tor/split', label)
    #     if not os.path.exists(class_dir):
    #         print(f"[跳过] 不存在目录: {class_dir}")
    #         continue
    #
    #     for dirpath, _, filenames in os.walk(class_dir):
    #         for filename in filenames:
    #             if filename.endswith(".pcap"):
    #                 full_path = os.path.join(dirpath, filename)
    #                 print(f"[处理] {full_path}  按 {packets_num} 包切分")
    #
    #                 # 切分命令：输出到当前文件所在目录
    #                 cmd = f'SplitCap -r "{full_path}" -s packets {packets_num} -o "{dirpath}"'
    #                 subprocess.run(cmd, shell=True)
    #
    #                 # 删除原始文件
    #                 os.remove(full_path)
    #                 print(f"[删除] 原始文件: {full_path}")

    class_dir = 'E:/ChromeDownload/Tor/split/FileTransfer'
    for dirpath, _, filenames in os.walk(class_dir):
        for filename in filenames:
            if filename.endswith(".pcap"):
                full_path = os.path.join(dirpath, filename)
                print(f"[处理] {full_path}  按 300 包切分")

                # 切分命令：输出到当前文件所在目录
                cmd = f'SplitCap -r "{full_path}" -s packets {300} -o "{dirpath}"'
                subprocess.run(cmd, shell=True)

                # 删除原始文件
                os.remove(full_path)
                print(f"[删除] 原始文件: {full_path}")