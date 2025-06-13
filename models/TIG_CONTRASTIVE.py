import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn

class TIG_SimCLR(torch.nn.Module):
    def __init__(self, config):
        super(TIG_SimCLR, self).__init__()
        self.config = config
        self.num_node_features = config.hyperparameters.num_node_features
        self.num_classes = config.dataset.num_classes if config.task.binary == False else 2
        self.hidden_size = config.hyperparameters.hidden_size
        self.projection_dim = getattr(config.hyperparameters, 'projection_dim', 128)
        
        # 添加输入投影层，将节点特征映射到hidden_size
        self.input_projection = nn.Linear(self.num_node_features, self.hidden_size)

        self.encoder = torch_geometric.nn.Sequential('x, edge_index, batch',
            [
                (torch_geometric.nn.GCNConv(self.hidden_size, self.hidden_size), 'x, edge_index -> x1'),
                (torch.nn.ReLU(inplace=True)),
                (torch.nn.Dropout(config.hyperparameters.dropout)),
                
                (torch_geometric.nn.GCNConv(self.hidden_size, self.hidden_size), 'x1, edge_index -> x2'),
                (torch.nn.ReLU(inplace=True)),
                (torch.nn.Dropout(config.hyperparameters.dropout)),

                (torch_geometric.nn.GCNConv(self.hidden_size, self.hidden_size), 'x2, edge_index -> x3'),
                (torch.nn.ReLU(inplace=True)),

                (lambda x1, x2, x3: [x1, x2, x3], 'x1, x2, x3 -> xs'),

                (torch_geometric.nn.models.JumpingKnowledge(mode='cat'), 'xs -> x'),
                (torch_geometric.nn.global_mean_pool, 'x, batch -> x')
            ]
        )
        
        # 投影头用于对比学习
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),  # JumpingKnowledge concatenation
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.projection_dim)
        )

    def forward(self, x, edge_index, batch, return_embedding=False):
        # 投影输入特征到hidden_size
        x = self.input_projection(x)
        # 通过编码器获得图级表示
        graph_embedding = self.encoder(x, edge_index, batch)
        
        if return_embedding:
            # 返回编码器的原始输出（用于微调）
            return graph_embedding
        else:
            # 通过投影头并归一化（用于对比学习）
            x = self.projection_head(graph_embedding)
            x = F.normalize(x, dim=1)
            return x

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: 两个增强视图的表示 [batch_size, projection_dim]
        """
        batch_size = z_i.shape[0]

        # 计算相似度矩阵
        representations = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, projection_dim]
        representations = F.normalize(representations, dim=1)  # 归一化
        similarity_matrix = torch.matmul(representations, representations.T)  # [2*batch_size, 2*batch_size]
        
        # 应用温度参数
        similarity_matrix = similarity_matrix / self.temperature
        
        # 创建标签：对角线上的正样本对
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(similarity_matrix.device)
        
        # 移除对角线（自身相似度）
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(similarity_matrix.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # 计算正样本的相似度
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        
        # 计算负样本的相似度
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        # InfoNCE损失
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


class TIG_CONTRASTIVE(nn.Module):
    def __init__(self, config):
        super(TIG_CONTRASTIVE, self).__init__()
        self.config = config
        self.encoder = TIG_SimCLR(config)
        self.loss_fn = ContrastiveLoss(temperature=getattr(config.hyperparameters, 'temperature', 0.1))
        
        # 添加分类头用于微调
        self.num_classes = config.dataset.num_classes if not config.task.binary else 2
        self.classifier = nn.Sequential(
            nn.Linear(config.hyperparameters.hidden_size * 3, config.hyperparameters.hidden_size),
            nn.ReLU(),
            nn.Dropout(getattr(config.hyperparameters, 'dropout', 0.5)),
            nn.Linear(config.hyperparameters.hidden_size, self.num_classes)
        )
        
    def forward(self, data1, data2=None, mode='pretrain'):
        """
        mode: 'pretrain' for contrastive learning, 'finetune' for supervised learning
        """
        if mode == 'pretrain':
            # 对比学习模式
            z1 = self.encoder(data1.x, data1.edge_index, data1.batch)
            z2 = self.encoder(data2.x, data2.edge_index, data2.batch)
            loss = self.loss_fn(z1, z2)
            return loss, z1, z2
        
        elif mode == 'finetune':
            # 微调模式 - 获取图级嵌入并通过分类器
            graph_embedding = self.encoder(data1.x, data1.edge_index, data1.batch, return_embedding=True)
            logits = self.classifier(graph_embedding)
            return logits
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def encode(self, x, edge_index, batch):
        """用于获取图的表示"""
        return self.encoder(x, edge_index, batch, return_embedding=True)



