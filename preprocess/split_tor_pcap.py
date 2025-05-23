import os
import sys
import subprocess
import shutil
from pathlib import Path
import logging
from datetime import datetime

# ==================== 配置区域 ====================
INPUT_DIR = "E:/ChromeDownload/Tor/Tor"  # PCAP文件输入目录
OUTPUT_DIR = "E:/ChromeDownload/Tor/filtered"  # 会话文件输出目录
SPLITCAP_PATH = None  # SplitCap路径(None=自动查找)
ENABLE_CLEANING = True  # 是否启用数据清洗
MAX_PACKETS_PER_SESSION = 500  # 每个会话最大包数
MAX_SIZE_PER_SESSION = "5MB"  # 每个会话最大大小


# 如果自动查找SplitCap失败，请取消注释并指定完整路径：
# SPLITCAP_PATH = "C:/Program Files/SplitCap/SplitCap.exe"
# ===================================================

class SimpleTorSplitter:
    def __init__(self):
        self.input_dir = Path(INPUT_DIR)
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'process.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # 查找SplitCap
        self.splitcap_path = self._find_splitcap()

    def _find_splitcap(self):
        """查找SplitCap可执行文件"""
        if SPLITCAP_PATH and os.path.exists(SPLITCAP_PATH):
            self.logger.info(f"使用指定的SplitCap: {SPLITCAP_PATH}")
            return SPLITCAP_PATH

        # 常见安装位置
        locations = [
            "SplitCap.exe",
            "./SplitCap.exe",
            "C:/Program Files/SplitCap/SplitCap.exe",
            "C:/Program Files (x86)/SplitCap/SplitCap.exe",
            "D:/Program Files/SplitCap/SplitCap.exe",
        ]

        # 检查PATH
        if shutil.which("SplitCap.exe"):
            path = shutil.which("SplitCap.exe")
            self.logger.info(f"在PATH中找到SplitCap: {path}")
            return path

        # 检查常见位置
        for path in locations:
            if os.path.exists(path):
                self.logger.info(f"找到SplitCap: {path}")
                return path

        self.logger.warning("未找到SplitCap，将使用默认路径")
        return "SplitCap.exe"

    def clean_pcap(self, pcap_file):
        """清洗PCAP文件"""
        if not ENABLE_CLEANING or not shutil.which("tshark"):
            return str(pcap_file)

        cleaned_file = self.output_dir / f"cleaned_{Path(pcap_file).name}"

        cmd = [
            "tshark", "-r", str(pcap_file), "-w", str(cleaned_file), "-Y",
            "(tcp.port == 9001 or tcp.port == 9030 or tcp.port == 9050 or "
            "tcp.port == 9051 or tcp.port == 9150 or tcp.port == 9151 or "
            "tcp.port == 443) and not dns and not dhcp"
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            if cleaned_file.exists() and cleaned_file.stat().st_size > 0:
                self.logger.info(f"清洗完成: {Path(pcap_file).name}")
                return str(cleaned_file)
        except Exception as e:
            self.logger.warning(f"清洗失败，使用原文件: {e}")

        return str(pcap_file)

    def split_sessions(self, pcap_file):
        """分隔会话"""
        pcap_path = Path(pcap_file)
        session_dir = self.output_dir / f"sessions_{pcap_path.stem}_{datetime.now().strftime('%H%M%S')}"
        session_dir.mkdir(exist_ok=True)

        # 清洗文件
        cleaned_file = self.clean_pcap(pcap_file)

        # SplitCap命令
        cmd = [
            self.splitcap_path,
            "-r", cleaned_file,
            "-o", str(session_dir) + "\\",
            "-s", "session",
            "-p", str(MAX_PACKETS_PER_SESSION),
            "-b", MAX_SIZE_PER_SESSION
        ]

        try:
            self.logger.info(f"分隔会话: {pcap_path.name}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)

            # 统计生成的文件
            session_files = list(session_dir.glob("*.pcap")) + list(session_dir.glob("*.cap"))
            self.logger.info(f"生成 {len(session_files)} 个会话文件")
            return session_files

        except subprocess.CalledProcessError as e:
            self.logger.error(f"SplitCap失败: {e}")
            return []
        except Exception as e:
            self.logger.error(f"处理失败: {e}")
            return []

    def process_all(self):
        """处理所有PCAP文件"""
        if not self.input_dir.exists():
            self.logger.error(f"输入目录不存在: {self.input_dir}")
            return

        # 查找PCAP文件
        pcap_files = []
        for pattern in ["*.pcap", "*.cap", "*.pcapng"]:
            pcap_files.extend(self.input_dir.glob(pattern))
            pcap_files.extend(self.input_dir.rglob(pattern))  # 包括子目录

        pcap_files = list(set(pcap_files))  # 去重

        if not pcap_files:
            self.logger.warning("未找到PCAP文件")
            return

        self.logger.info(f"找到 {len(pcap_files)} 个PCAP文件")

        # 处理每个文件
        total_sessions = 0
        success_count = 0

        for i, pcap_file in enumerate(pcap_files, 1):
            self.logger.info(f"进度 {i}/{len(pcap_files)}: {pcap_file.name}")

            try:
                session_files = self.split_sessions(pcap_file)
                total_sessions += len(session_files)
                success_count += 1
            except Exception as e:
                self.logger.error(f"处理 {pcap_file.name} 失败: {e}")

        # 生成报告
        self._generate_report(success_count, len(pcap_files), total_sessions)

    def _generate_report(self, success, total, sessions):
        """生成处理报告"""
        report = self.output_dir / "processing_report.txt"
        with open(report, 'w', encoding='utf-8') as f:
            f.write("Tor流量处理报告\n")
            f.write("=" * 30 + "\n")
            f.write(f"处理时间: {datetime.now()}\n")
            f.write(f"输入目录: {self.input_dir}\n")
            f.write(f"输出目录: {self.output_dir}\n")
            f.write(f"成功处理: {success}/{total} 个文件\n")
            f.write(f"总会话数: {sessions}\n")

        self.logger.info(f"报告已生成: {report}")


def main():
    """主函数"""
    print("=" * 50)
    print("Tor流量会话分隔工具 (简化版)")
    print("=" * 50)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"数据清洗: {'启用' if ENABLE_CLEANING else '禁用'}")
    print("-" * 50)

    try:
        splitter = SimpleTorSplitter()
        splitter.process_all()
        print("\n处理完成! 请查看输出目录和日志文件。")

    except KeyboardInterrupt:
        print("\n用户中断处理")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()