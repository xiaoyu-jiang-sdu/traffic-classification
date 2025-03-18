def get_link_name(link_type_id):
    link_type_mapping = {
        0: "NULL",  # BSD loopback encapsulation
        1: "ETHERNET",  # Ethernet (10Mb, 100Mb, 1000Mb, and up)
        6: "IEEE802_5",  # Token Ring
        7: "ARCNET",  # ARCNET
        8: "SLIP",  # Serial Line IP
        9: "PPP",  # Point-to-Point Protocol
        10: "FDDI",  # FDDI
        50: "PPP_HDLC",  # PPP in HDLC-like framing
        101: "RAW",  # Raw IP
        113: "SITA",  # SITA
        114: "ERF",  # Endace ERF
        127: "LINUX_LAPD",  # Linux Lapd
        228: "DECT",  # DECT
        229: "AOE",  # ATA over Ethernet
        230: "Z_WAVE_R1_R2",  # Z-Wave R1 and R2
        231: "Z_WAVE_R3",  # Z-Wave R3
        147: "BLUETOOTH_H4",  # Bluetooth HCI UART transport layer
        148: "USB_LINUX",  # USB packets, beginning with a Linux USB header
        240: "IEEE802_15_4",  # IEEE 802.15.4
        247: "USB_PCAP",  # USB packets captured with USBPcap
    }
    return link_type_mapping[link_type_id]
