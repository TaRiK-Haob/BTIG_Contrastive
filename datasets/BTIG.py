import torch
import itertools
from torch_geometric.data import Data
import numpy as np

#* 添加突发内边
def _get_intra_edge(burst_data):
    result = []
    for i in range(1, len(burst_data)):
        result.append((burst_data[i-1],burst_data[i]))
        result.append((burst_data[i],burst_data[i-1]))
    return result

#* 添加突发间边
def _get_inter_edge(burst1, burst2, fully_connect):
    result = []
    if fully_connect:
        for i in burst1:
            for j in burst2:
                result += list(itertools.permutations([i]+[j], 2))
    else:
        result = list(itertools.permutations([burst1[0], burst2[0]], 2)) + list(itertools.permutations([burst1[-1], burst2[-1]], 2))
    return list(set(result))

def construct_BTIG(pkt_direct, pkt_arrtime, pkt_feat, label = -1, burst_threshold = 1, fully_connect = False):

    node_feat = []
    node = []
    y = -1

    #* Add vertices and corresponding node features
    for i, f in enumerate(pkt_feat):
        #* 包长作为节点特征
        # node_feat.append([pkt_feat[i], i])
        node_feat.append([pkt_feat[i]])
        node.append(i)

    #* Divide into burst
    burst_set = []
    burst = [0]

    # Burst
    for i in range(1, len(pkt_direct)):
        if (pkt_direct[i] == pkt_direct[i-1]) and (pkt_arrtime[i] - pkt_arrtime[i - 1] < burst_threshold):
            burst.append(i)
        else:
            burst_set.append(burst)
            burst = [i]
    burst_set.append(burst)

    # print(burst_set)

    # ==================ADD Burst Embedding================
    for index, b in enumerate(burst_set):
        for pkt in b:
            node_feat[pkt].append(index)
            node_feat[pkt].append(len(b))
    # =====================================================

    #* Add intra-burst edges and inter-burst edges
    #* intra-burst:
    edge_pair1 = []
    for b in burst_set:
        if len(b) > 1:
            edge_pair1 += _get_intra_edge(b)

    #* inter-burst:
    edge_pair2 = []
    for i in range(1, len(burst_set)):
        front_b = burst_set[i-1]
        rear_b = burst_set[i]
        edge_pair2 += _get_inter_edge(front_b, rear_b, fully_connect)

    #* 将edge_pair转化为COO格式
    edge_pair = edge_pair1 + edge_pair2
    src = [edge[0] for edge in edge_pair]
    dst = [edge[1] for edge in edge_pair]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    #* 将node和node_feat转化为x
    x = torch.tensor([x for x in node_feat], dtype=torch.float)
    y = torch.tensor([label])

    # 添加统计特征
    vector = []

    # 处理时间间隔
    time_intervals = [pkt_arrtime[i] - pkt_arrtime[i-1] for i in range(1, len(pkt_arrtime))]
    IAT_Mean = np.mean(time_intervals) if time_intervals else 0
    IAT_Std = np.std(time_intervals) if time_intervals else 0
    
    # 处理前向包
    Foward = [pkt_arrtime[i] for i in range(len(pkt_arrtime)) if pkt_direct[i] == 1]
    F_intervals = [Foward[i] - Foward[i-1] for i in range(1, len(Foward))]
    F_IAT_Mean = np.mean(F_intervals) if F_intervals else 0
    F_IAT_Std = np.std(F_intervals) if F_intervals else 0

    # 处理后向包
    Backward = [pkt_arrtime[i] for i in range(len(pkt_arrtime)) if pkt_direct[i] == -1]
    B_intervals = [Backward[i] - Backward[i-1] for i in range(1, len(Backward))]
    B_IAT_Mean = np.mean(B_intervals) if B_intervals else 0
    B_IAT_Std = np.std(B_intervals) if B_intervals else 0

    # pkt_feat = np.abs(pkt_feat)
    Foward = [i for i in pkt_feat if i > 0]
    Backward = [abs(i) for i in pkt_feat if i < 0]

    pkt_len_Mean = np.mean(pkt_feat)
    pkt_len_Std = np.std(pkt_feat)

    B_pkt_len_Mean = np.mean(Backward) if Backward else 0
    B_pkt_len_Std = np.std(Backward) if Backward else 0

    F_pkt_len_Mean = np.mean(Foward) if Foward else 0
    F_pkt_len_Std = np.std(Foward)  if Foward else 0

    vector = [IAT_Mean, IAT_Std, B_IAT_Mean, B_IAT_Std, F_IAT_Mean, F_IAT_Std, 
              pkt_len_Mean, pkt_len_Std, B_pkt_len_Mean, B_pkt_len_Std, F_pkt_len_Mean, F_pkt_len_Std]

    vector = torch.tensor(vector, dtype=torch.float)

    data = Data(x = x, edge_index = edge_index, y = y, vector = vector)

    #* return
    return data

def construct_BTIG_wo_statistical(pkt_direct, pkt_arrtime, pkt_feat, label = -1, burst_threshold = 1, fully_connect = False):

    node_feat = []
    node = []
    y = -1

    #* Add vertices and corresponding node features
    for i, f in enumerate(pkt_feat):
        #* 包长作为节点特征
        # node_feat.append([pkt_feat[i], i])
        node_feat.append([pkt_feat[i]])
        node.append(i)

    #* Divide into burst
    burst_set = []
    burst = [0]

    # Burst
    for i in range(1, len(pkt_direct)):
        if (pkt_direct[i] == pkt_direct[i-1]) and (pkt_arrtime[i] - pkt_arrtime[i - 1] < burst_threshold):
            burst.append(i)
        else:
            burst_set.append(burst)
            burst = [i]
    burst_set.append(burst)

    # print(burst_set)

    # ==================ADD Burst Embedding================
    for index, b in enumerate(burst_set):
        for pkt in b:
            node_feat[pkt].append(index)
            node_feat[pkt].append(len(b))
    # =====================================================

    #* Add intra-burst edges and inter-burst edges
    #* intra-burst:
    edge_pair1 = []
    for b in burst_set:
        if len(b) > 1:
            edge_pair1 += _get_intra_edge(b)

    #* inter-burst:
    edge_pair2 = []
    for i in range(1, len(burst_set)):
        front_b = burst_set[i-1]
        rear_b = burst_set[i]
        edge_pair2 += _get_inter_edge(front_b, rear_b, fully_connect)

    #* 将edge_pair转化为COO格式
    edge_pair = edge_pair1 + edge_pair2
    src = [edge[0] for edge in edge_pair]
    dst = [edge[1] for edge in edge_pair]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    #* 将node和node_feat转化为x
    x = torch.tensor([x for x in node_feat], dtype=torch.float)
    y = torch.tensor([label])

    data = Data(x = x, edge_index = edge_index, y = y)

    #* return
    return data