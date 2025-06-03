import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,GCNConv
import torch
from torch import cat
import torch.nn.functional as F

class GCN_multi_scale(torch.nn.Module):
    def __init__(self, config):
        super(GCN_multi_scale, self).__init__()
        self.config = config
        self.num_statistical_features = config.num_statistical_features
        self.num_node_features = config.num_node_features
        self.num_classes = config.num_classes if config.binary == False else 2
        self.hidden_size = config.hidden_size
        
        # 修正GAT层头数和维度
        self.agg1 = GCNConv(self.num_node_features, self.hidden_size)
        self.agg2 = GCNConv(self.hidden_size, self.hidden_size)
        self.agg3 = GCNConv(self.hidden_size, self.hidden_size)
        
        # 统计特征处理
        self.statistical_nn = torch.nn.Linear(self.num_statistical_features, self.hidden_size)
        
        # 合并层
        self.combine = torch.nn.Linear(self.hidden_size * 4, self.num_classes)

    def forward(self, x, edge_index, statistical, batch):
        # print(f"Input x shape: {x.shape}")
        x1 = x = self.agg1(x, edge_index)
        x1 = F.relu(x1)

        x2 = x = self.agg2(x, edge_index)
        x2 = F.relu(x2)

        x3 = x = self.agg3(x, edge_index)
        x3 = F.relu(x3)

        # 全局池化
        x1 = global_mean_pool(x1, batch)
        x2 = global_mean_pool(x2, batch)
        x3 = global_mean_pool(x3, batch)

        x1x2x3 = cat([x1, x2, x3], dim = 1)

        # 统计特征处理
        stastatisticaltis = statistical.reshape(-1, self.num_statistical_features)
        statistical = self.statistical_nn(statistical)
        statistical = F.relu(statistical)
        
        # 特征合并
        x = torch.cat([x1x2x3, statistical], dim = 1)
        x = self.combine(x)
        
        return x