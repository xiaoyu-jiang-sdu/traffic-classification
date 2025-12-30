import os
from appscanner.preprocessor import Preprocessor

base_dir = r"E:/ChromeDownload/USTC-TFC2016-master/flows_sampled"

pcap_paths = []
labels = []

for label in os.listdir(base_dir):
    label_dir = os.path.join(base_dir, label)
    if not os.path.isdir(label_dir):
        continue

    for file in os.listdir(label_dir):
        if file.endswith(".pcap"):
            pcap_paths.append(os.path.join(label_dir, file))
            labels.append(label)