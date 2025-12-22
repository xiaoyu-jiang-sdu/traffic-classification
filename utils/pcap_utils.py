import os
import subprocess
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


# 按照时间间隔进行划分, 默认1s
# Tor划分
def split_cap_time(pcap_file, output_dir, seconds=1):
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"SplitCap -r \"{pcap_file}\" -s seconds {seconds} -o \"{output_dir}\""

    # 执行命令
    os.system(cmd)
# def split_pcap_by_fixed_time_interval(pcap_file, output_dir, interval=1):
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 提取所有时间戳
#     cmd = f'tshark -r "{pcap_file}" -T fields -e frame.time_epoch'
#     result = subprocess.check_output(cmd, shell=True).decode()
#     times = [float(t) for t in result.splitlines() if t.strip()]
#     if not times:
#         print("无有效包")
#         return
#
#     start_time = int(times[0])
#     end_time = int(times[-1])
#     print(f"划分时间：{start_time}s ~ {end_time}s，每 {interval}s 一段")
#
#     # 遍历时间段，按窗口切分
#     slice_id = 0
#     for t_start in range(start_time, end_time, interval):
#         t_end = t_start + interval
#         output_file = os.path.join(output_dir, f"time_{slice_id:05d}.pcap")
#         filter_str = f'frame.time_epoch >= {t_start} && frame.time_epoch < {t_end}'
#         cmd = f'tshark -r "{pcap_file}" -Y "{filter_str}" -w "{output_file}"'
#         subprocess.call(cmd, shell=True)
#         slice_id += 1
def split_pcap_by_flow_tshark(pcap_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 生成每个 TCP 会话编号
    cmd = f"tshark -r \"{pcap_file}\" -T fields -e tcp.stream"
    result = subprocess.check_output(cmd, shell=True).decode()
    streams = sorted(set(line.strip() for line in result.splitlines() if line.strip().isdigit()))

    for stream_id in streams:
        out_file = os.path.join(output_dir, f"flow_{stream_id}.pcap")
        extract_cmd = (
            f"tshark -r \"{pcap_file}\" -Y \"tcp.stream == {stream_id}\" "
            f"-w \"{out_file}\""
        )
        subprocess.call(extract_cmd, shell=True)


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
    dir = "E:/ChromeDownload/Tor/filtered"
    filename = "Torrent01.pcapng"
    convert_pcapng_2_pcap(dir, filename, dir)
