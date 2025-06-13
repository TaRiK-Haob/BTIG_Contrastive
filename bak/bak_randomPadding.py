# It pads a random amount of data to packet payloads under the restriction that the packet size is less than the Maximum Transmission Unit (MTU).
# Data overhead means the sizes of inserted and padded packets divided by the total size of packets.
import random

def random_padding(flow, max_nodes = 128, probability = 0.2):
    '''
    输入：flow，概率阈值probability，max_nodes

    输出：padding_flow
    '''

    pkt_len_seq = flow['pkt_len_seq']
    pkt_ts_seq = flow['pkt_ts_seq']

    if not len(pkt_len_seq) == len(pkt_ts_seq):
        print("The length of pkt_len_seq and pkt_ts_seq are not equal!")
        return
    
    len_flow = len(pkt_len_seq)

    if len_flow > max_nodes:
        pkt_len_seq = pkt_len_seq[:max_nodes]
        pkt_ts_seq = pkt_ts_seq[:max_nodes]

    max_len = max([i for i in pkt_len_seq])
    min_len = min([i for i in pkt_len_seq])

    result = {"pkt_len_seq": [],
              "pkt_ts_seq": []}

    i = 0
    while(i < len(pkt_len_seq)):
        q = random.uniform(0, 1)
        # 计算当前包的大小
        if q < probability:
            # 计算填充数据的大小
            padding_size = random.randint(0, 1500)
            # 计算填充数据的时间戳
            padding_time = random.uniform(0, 0.2)
            # 更新包的大小和时间戳
            padding_direction = random.choice([-1, 1])

            result['pkt_len_seq'].append(padding_direction * padding_size)
            result['pkt_ts_seq'].append(pkt_ts_seq[i] + padding_time)

        else:
            result['pkt_len_seq'].append(pkt_len_seq[i])
            result['pkt_ts_seq'].append(pkt_ts_seq[i])
        i += 1

    flow['pkt_len_seq'] = result['pkt_len_seq']
    flow['pkt_ts_seq'] = result['pkt_ts_seq']
    # 如果填充后的数据长度超过max_nodes，则截断
    if len(flow['pkt_len_seq']) > max_nodes:
        flow['pkt_len_seq'] = flow['pkt_len_seq'][:max_nodes]
        flow['pkt_ts_seq'] = flow['pkt_ts_seq'][:max_nodes]

    return flow

if __name__ == '__main__':
    flow = {"pkt_len_seq": [52, -48, 40, 557, -40, -1500, -1500, -2306, 40, 166, 93, 96, 82, 887, -298, 40, -40, -78, 78, -40, -1500, -1500, -1157, -376, 40, 40, -40, 40], "pkt_ts_seq": [0.0, 0.0288, 0.0289, 0.029, 0.0578, 0.0599, 0.0602, 0.0603, 0.0604, 0.0623, 0.0627, 0.0627, 0.0627, 0.0629, 0.0911, 0.0913, 0.0915, 0.0915, 0.0915, 0.1203, 0.1299, 0.1302, 0.1302, 0.1302, 0.1303, 2.6747, 2.7035, 2.7036], "proto": 6, "label": 0, "class": 0}
    print('原始：\t\t', flow['pkt_len_seq'])
    # 测试
    flow = random_padding(flow)
    print('padding后\t', flow['pkt_len_seq'])