import torch
import torch.nn as nn
import time
import hydra
import logging
from models import get_model
from datasets import get_dataloader
import random
import json
import itertools

logger = logging.getLogger(__name__)

class Tester:
    def __init__(self, config):
        self.config = config
        self.model = get_model(config)
        self.dataloader = get_dataloader(config)
        self.device = config.device

    def load_testAndfinetune_data(self):
        """加载测试数据集"""
        logger.info("Loading test & fine-tune data...")
        data_idx, data_x, data_y = [],[],[]
        labels = ['benign']
        with open(self.config.tester.test_data_path, 'rb') as f:
            for idx,line in enumerate(f):
                data = json.loads(line)
                if data['class'] not in labels:
                    labels.append(data['class'])
                data_idx.append(idx)
                data_x.append(torch.tensor(self.pad_sequence([1 if x_i > 0 else -1 for x_i in data['pkt_len_seq']], 5000, pad_value=0), dtype=torch.float32).unsqueeze(0))  # 转换为tensor并添加维度
                data_y.append(labels.index(data['class']))

        return list(zip(data_idx, data_x, data_y))

    def test(self):
        self.model.load_state_dict(torch.load(self.config.output_settings.best_model_path))
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        average_loss = total_loss / total_samples if total_samples > 0 else 0.0
        logger.info(f"Test Loss: {average_loss:.4f}")



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