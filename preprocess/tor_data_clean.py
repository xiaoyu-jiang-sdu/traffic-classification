import os
import glob


def find_files(directory, extension="*.pcap"):
    """查找指定目录下的pcap文件"""
    pattern = os.path.join(directory, "**", extension)
    return glob.glob(pattern, recursive=True)


def simple_tor_filter(pcap_dir, output_dir="/filtered"):
    """
    简化版Tor流量过滤，使用基础语法
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pcap_files = find_files(pcap_dir)

    if not pcap_files:
        print(f"在目录 {pcap_dir} 中未找到pcap文件")
        return

    for pcap_file in pcap_files:
        filename = os.path.basename(pcap_file)
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{base_name}_simple_filtered.pcap")

        # 使用最简单的过滤规则
        simple_filter = "tcp and (tcp.port == 9001 or tcp.port == 9030 or tcp.port == 9050 or tcp.port == 9051 or tcp.port == 80 or tcp.port == 443)"

        print(f"正在处理文件: {filename}")
        command = f'tshark -r "{pcap_file}" -Y "{simple_filter}" -w "{output_file}"'

        print(f"执行命令: {command}")  # 调试信息
        result = os.system(command)

        if result == 0:
            if os.path.getsize(output_file) > 24:  # 检查文件不为空
                print(f"✓ 简单过滤完成: {base_name}_simple_filtered.pcap")
            else:
                print(f"- 无匹配流量: {filename}")
        else:
            print(f"✗ 过滤失败: {filename}")


def filter_tor_traffic(pcap_dir, output_dir="/filtered"):
    """
    使用tshark对Tor流量进行过滤和清洗
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pcap_files = find_files(pcap_dir)

    if not pcap_files:
        print(f"在目录 {pcap_dir} 中未找到pcap文件")
        return

    # Tor流量过滤规则 - 使用正确的tshark语法
    tor_filters = [
        # 基础Tor端口过滤
        "tcp and (tcp.port == 9001 or tcp.port == 9030 or tcp.port == 9050 or tcp.port == 9051)",
        # 常见的Tor目录服务器端口
        "tcp and (tcp.port == 80 or tcp.port == 443)",
        # OR端口范围 (Onion Router ports)
        "tcp and (tcp.port >= 9001 and tcp.port <= 9150)",
        # 排除本地回环和广播
        "not (ip.src == 127.0.0.1 or ip.dst == 127.0.0.1 or ip.dst == 255.255.255.255)"
    ]

    # 组合过滤规则
    combined_filter = f"(({tor_filters[0]}) or ({tor_filters[1]}) or ({tor_filters[2]})) and ({tor_filters[3]})"

    for pcap_file in pcap_files:
        filename = os.path.basename(pcap_file)
        base_name = os.path.splitext(filename)[0]

        # 原始过滤文件
        filtered_file = os.path.join(output_dir, f"{base_name}_filtered.pcap")

        # 执行基础Tor流量过滤
        print(f"正在处理文件: {filename}")
        command = f'tshark -r "{pcap_file}" -Y "{combined_filter}" -w "{filtered_file}"'
        result = os.system(command)

        if result == 0:
            # 进一步清洗：移除重复和无效包
            cleaned_file = os.path.join(output_dir, f"{base_name}_cleaned.pcap")

            # 去重和大小过滤
            clean_command = f'tshark -r "{filtered_file}" -Y "frame.len > 64 and tcp.len > 0" -w "{cleaned_file}"'
            clean_result = os.system(clean_command)

            if clean_result == 0:
                print(f"✓ 清洗完成: {base_name}_cleaned.pcap")

                # 可选：删除中间文件以节省空间
                # os.remove(filtered_file)

                # 生成统计信息
                generate_stats(cleaned_file, output_dir, base_name)
            else:
                print(f"✗ 清洗失败: {filename}")
        else:
            print(f"✗ 过滤失败: {filename}")


def generate_stats(pcap_file, output_dir, base_name):
    """生成流量统计信息"""
    stats_file = os.path.join(output_dir, f"{base_name}_stats.txt")

    # 基本统计
    stat_commands = [
        f'tshark -r "{pcap_file}" -q -z conv,tcp > "{stats_file}"',
        f'echo "\n=== 端口分布 ===" >> "{stats_file}"',
        f'tshark -r "{pcap_file}" -T fields -e tcp.dstport | sort | uniq -c | sort -nr | head -20 >> "{stats_file}"',
        f'echo "\n=== 包数量统计 ===" >> "{stats_file}"',
        f'tshark -r "{pcap_file}" -q -z io,stat,1 >> "{stats_file}"'
    ]

    for cmd in stat_commands:
        os.system(cmd)


def advanced_tor_filter(pcap_dir, output_dir="/filtered"):
    """
    高级Tor流量过滤，包含更精细的规则
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pcap_files = find_files(pcap_dir)

    # 更精细的Tor流量特征 - 修正语法
    advanced_filters = {
        'directory_requests': 'tcp and tcp.port == 80 and http.request.uri contains "tor/status"',
        'or_connections': 'tcp and (tcp.port == 9001 or tcp.port == 9030) and tcp.flags.syn == 1',
        'control_connections': 'tcp and tcp.port == 9051',
        'socks_proxy': 'tcp and tcp.port == 9050',
        'bridge_traffic': 'tcp and (tcp.port >= 9001 and tcp.port <= 9150) and tcp.window_size_value > 8192',
        'encrypted_streams': 'tcp and tcp.len > 100 and not http and not dns',
        'tor_range_ports': 'tcp and (tcp.port >= 9001 and tcp.port <= 9150)',
        'https_connections': 'tcp and tcp.port == 443'
    }

    for pcap_file in pcap_files:
        filename = os.path.basename(pcap_file)
        base_name = os.path.splitext(filename)[0]

        print(f"正在进行高级处理: {filename}")

        # 为每种类型创建单独文件
        for filter_name, filter_rule in advanced_filters.items():
            output_file = os.path.join(output_dir, f"{base_name}_{filter_name}.pcap")
            command = f'tshark -r "{pcap_file}" -Y "{filter_rule}" -w "{output_file}"'

            result = os.system(command)
            if result == 0:
                # 检查文件是否为空
                if os.path.getsize(output_file) > 24:  # pcap header size
                    print(f"  ✓ {filter_name}: 提取完成")
                else:
                    os.remove(output_file)
                    print(f"  - {filter_name}: 无匹配流量")
            else:
                print(f"  ✗ {filter_name}: 提取失败")


if __name__ == '__main__':
    # 配置路径
    input_dir = "E:/ChromeDownload/Tor/Tor"  # 输入目录
    output_dir = "/filtered"  # 输出目录

    print("开始Tor流量清洗...")
    print("=" * 50)

    # 先尝试简单过滤
    print("1. 执行简单Tor流量过滤...")
    simple_tor_filter(input_dir, output_dir)

    print("\n" + "=" * 50)

    # 如果简单过滤成功，再尝试基础过滤
    print("2. 执行基础Tor流量过滤...")
    try:
        filter_tor_traffic(input_dir, output_dir)
    except Exception as e:
        print(f"基础过滤失败: {e}")

    print("\n" + "=" * 50)

    # 高级过滤（可选）
    print("3. 执行高级分类过滤...")
    try:
        advanced_output_dir = os.path.join(output_dir, "advanced")
        advanced_tor_filter(input_dir, advanced_output_dir)
    except Exception as e:
        print(f"高级过滤失败: {e}")

    print("\n" + "=" * 50)
    print("Tor流量清洗完成！")
    print(f"输出目录: {output_dir}")