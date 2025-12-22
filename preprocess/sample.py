import os
import random
import shutil

from utils.pcap_utils import find_files


def sample_pcap(split_pcap_dir, output_dir='flows_sampled', maximum=6000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    files = find_files(split_pcap_dir)
    if len(files) > maximum:
        files = random.sample(files, maximum)
        for file in files:
            target_path = os.path.join(output_dir, os.path.basename(file))
            shutil.copy2(file, target_path)

        print(f"✅ 采样完成，共复制 {len(files)} 个文件到 {output_dir}")
    else:
        for file in files:
            target_path = os.path.join(output_dir, os.path.basename(file))
            shutil.copy2(file, target_path)

        print(f"✅ 采样完成，共复制 {len(files)} 个文件到 {output_dir}")

# def sample_test(split_pcap_dir, output_dir='test', maximum=500):
#     random.seed(114514)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)
#     files = find_files(split_pcap_dir)
#     if len(files) > maximum:
#         files = random.sample(files, maximum)
#         for file in files:
#             target_path = os.path.join(output_dir, os.path.basename(file))
#             shutil.copy2(file, target_path)
#
#         print(f"✅ 采样完成，共复制 {len(files)} 个文件到 {output_dir}")


if __name__ == '__main__':
    random.seed(2025)
    # path = "E:/ChromeDownload/Tor"
    # output_dir = os.path.join(path, "flows_sampled")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    #
    # dirs = os.listdir(os.path.join(path, "split"))
    # for dirname in dirs:
    #     dir_path = os.path.join(path, 'split',  dirname)
    #     if os.path.isdir(dir_path):
    #         sample_pcap(dir_path, os.path.join(output_dir, dirname), 1800)
    # path = "E:/ChromeDownload/USTC-TFC2016-master"
    # output_dir = os.path.join(path, "flows_sampled")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    #
    # dirs = os.listdir(os.path.join(path, "split"))
    # for dirname in dirs:
    #     dir_path = os.path.join(path, 'split',  dirname)
    #     if os.path.isdir(dir_path):
    #         sample_pcap(dir_path, os.path.join(output_dir, dirname))

    path = "E:/ChromeDownload/USTC-TFC2016-master/split/Facetime"
    output_dir = "E:/ChromeDownload/USTC-TFC2016-master/flows_sampled/Facetime"
    sample_pcap(path, output_dir)
