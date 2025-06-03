from torch_geometric.data import Dataset
import os
import json
import linecache
import torch
from . import BTIG

class BTIGDataset(Dataset):
    def __init__(self, config):
        super(BTIGDataset, self).__init__()

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

    def get(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of range")
        
        line = self._get_line(idx)

        pkt_len_seq = line['pkt_len_seq'][:self.max_nodes]
        pkt_ts_seq = line['pkt_ts_seq'][:self.max_nodes]
        
        g = BTIG.construct_BTIG([1 if x > 0 else -1 for x in pkt_len_seq],
                               pkt_ts_seq,
                                [(i // 8) for i in pkt_len_seq],
                                line['label'],
                                burst_threshold = self.burst_threshold)

        if self.binary and g.y:
            g.y = torch.tensor([1])
            
        if g.num_nodes <= self.max_nodes:
            return g
        else:
            subset = torch.tensor([i for i in range(self.max_nodes)])
            return g.subgraph(subset)