# 记录
### 预处理阶段
原始数据：pcap;pcapng
1. 将pcapng格式转换为pcap格式
2. 过滤冗余packet(只关注TCP,UDP.TLS?,需要删去TLS的负载？)
3. 切分、重新组织数据（2B大小的数据作为token。按照session分隔）
4. 每个token进入Embedding转为向量

token：每2B个数据,允许共享1B数据？（如1234 34ab）

序列T：packet/session/burst？预训练可以用flow来训练，
微调和预测关注的是packet???

burst: 将一个会话一定时间内相同方向的流量作为一个序列输入 ，
一个burst结束直到出现与当前流量方向不同的流量

### 预训练