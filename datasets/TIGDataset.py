from torch_geometric.data import Dataset, Data
import os
import json
import linecache
import torch
import itertools
import numpy as np
from my_utils.TrafficFlowAugmentation import TrafficFlowAugmentation
import my_utils.TrafficGraphAugmentation as TGA
import random

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

def _construct_TIG(pkt_direct, pkt_feat, label = -1):

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
        if (pkt_direct[i] == pkt_direct[i-1]):
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
        edge_pair2 += _get_inter_edge(front_b, rear_b, False)

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

class TIGDataset(Dataset):
    def __init__(self, config):
        super(TIGDataset, self).__init__()

        self.length = 0
        self.root = os.path.abspath(
            os.path.join(config.dataset.data_dir, config.dataset.data_name))

        with open(self.root) as f:
            self.length = sum(1 for _ in f)

        self.max_nodes = config.parameters.max_nodes
        self.binary = config.task.binary
        self.config_num_classes = config.dataset.num_classes
        self.config_num_node_features = config.hyperparameters.num_node_features
        self.config_num_statistical_features = config.hyperparameters.num_statistical_features
        self.burst_threshold = config.parameters.burst_threshold
        self.config = config

        self.aug_stage = config.obfuscation.stage


    def len(self):
        return self.length
    
    def _get_line(self, idx):
        try:
            line = linecache.getline(self.root, idx + 1).strip()
            if not line:  # 检查空行
                raise ValueError(f"Line {idx + 1} is empty")
            return json.loads(line)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误在第{idx + 1}行: {e}")
            print(f"问题行内容: {line}")
            raise
        except Exception as e:
            print(f"读取第{idx + 1}行时出错: {e}")
            raise

    @property
    def num_statistical_features(self) -> int:
        r"""Returns the number of statistical_features per node in the dataset."""
        return self.config_num_statistical_features
    
    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""

        return self.config_num_node_features
    
    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        if self.binary:
            return 2
        else:
            return self.config_num_classes

    def _build_graph(self, line):
        pkt_len_seq = line['pkt_len_seq'][:self.max_nodes]
        # pkt_ts_seq = line['pkt_ts_seq'][:self.max_nodes]
        
        g = _construct_TIG([1 if x > 0 else -1 for x in pkt_len_seq],
                                [(i // 8) for i in pkt_len_seq],
                                line['label'])
        if self.binary and g.y:
            g.y = torch.tensor([1])
            
        if g.num_nodes <= self.max_nodes:
            return g
        else:
            subset = torch.tensor([i for i in range(self.max_nodes)])
            return g.subgraph(subset)

    def get(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of range")
        
        # original sample
        line = self._get_line(idx)
        line_aug1 = line_aug2 = line

        # stage1 augmentation
        if self.config.obfuscation.stage == "stage1" or self.config.obfuscation.stage == "stage3":
            line_aug1 = TrafficFlowAugmentation(line)
            line_aug2 = TrafficFlowAugmentation(line)

        # stage2 augmentation
        if self.config.obfuscation.stage == "stage2" or self.config.obfuscation.stage == "stage3":
            line_aug1 = TGA.SubGraph(line_aug1)
            line_aug1 = TGA.SubGraph(line_aug2)

        g_aug1 = self._build_graph(line_aug1)
        g_aug2 = self._build_graph(line_aug2)

            # g_aug1 = TGA.feature_masking(self._build_graph(line_aug1))
            # g_aug2 = TGA.feature_masking(self._build_graph(line_aug2))

        # * Build the original graph and the augmented graph
        # g = self._build_graph(line)
        # g_aug1 = self._build_graph(line_aug1)
        # g_aug2 = self._build_graph(line_aug2)

        return g_aug1, g_aug2