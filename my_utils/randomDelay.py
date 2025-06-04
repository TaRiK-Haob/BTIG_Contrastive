# It randomly chooses and delays original packets from traffic flow by randomly generating a millisecond-level delay time.
# Time overhead is defined as the proportion of total delaying time to the original flow completion time.

import random

def random_delay(flow, max_nodes = 128, probability=0.2):
    '''
    输入：flow，概率阈值probability， max_nodes

    输出：delay_flow
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

    IAT = []
    for i in range(1, len(pkt_ts_seq)):
        IAT.append(pkt_ts_seq[i] - pkt_ts_seq[i-1])

    result = {"pkt_len_seq": pkt_len_seq,
              "pkt_ts_seq": []}
    
    i = 0
    while(i < len(IAT)):
        q = random.uniform(0, 1)
        # 计算当前包的大小
        if q < probability:
            # 计算延迟时间
            delay_time = random.uniform(0, 0.2)
            # 更新包的时间戳
            IAT[i] += delay_time
        else:
            IAT[i] = IAT[i]
        i += 1

    for i in range(len(pkt_ts_seq)):
        if i == 0:
            result['pkt_ts_seq'].append(0.0)
        else:
            delay = result['pkt_ts_seq'][i-1] + IAT[i-1]
            result['pkt_ts_seq'].append(delay)

    result['pkt_ts_seq'] = [round(x, 4) for x in result['pkt_ts_seq']]

    flow['pkt_len_seq'] = result['pkt_len_seq']
    flow['pkt_ts_seq'] = result['pkt_ts_seq']
    
    # 如果延迟后的数据长度超过max_nodes，则截断
    if len(flow['pkt_len_seq']) > max_nodes:
        flow['pkt_len_seq'] = flow['pkt_len_seq'][:max_nodes]
        flow['pkt_ts_seq'] = flow['pkt_ts_seq'][:max_nodes]

    return flow

if __name__ == '__main__':
    flow = {"pkt_len_seq": [52, -48, 40, 557, -40, -1500, -1500, -2306, 40, 166, 93, 96, 82, 887, -298, 40, -40, -78, 78, -40, -1500, -1500, -1157, -376, 40, 40, -40, 40], 
            "pkt_ts_seq": [0.0, 0.0288, 0.0289, 0.029, 0.0578, 0.0599, 0.0602, 0.0603, 0.0604, 0.0623, 0.0627, 0.0627, 0.0627, 0.0629, 0.0911, 0.0913, 0.0915, 0.0915, 0.0915, 0.1203, 0.1299, 0.1302, 0.1302, 0.1302, 0.1303, 2.6747, 2.7035, 2.7036], "proto": 6, "label": 0, "class": 0}
    print('原始：\t',flow['pkt_ts_seq'])
    # 测试
    flow = random_delay(flow)
    print('delay后\t',flow['pkt_ts_seq'])