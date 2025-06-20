import random
import torch_geometric.data
import torch

def SubGraph(line):
    """
    Augments a traffic graph by applying random feature transformations.
    """
    num = len(line['pkt_len_seq'])

    if num <= 4:
        return line

    q = random.uniform(0.8, 0.9)
    num = int(num * q)

    line['pkt_len_seq'] = line['pkt_len_seq'][:num]
    line['pkt_ts_seq'] = line['pkt_ts_seq'][:num]

    # result = _feature_masking(result)

    return line


def feature_masking(graph: torch_geometric.data.Data, probability = 0.2):
    """
    Randomly masks some features of the nodes in the graph.
    
    Args:
        graph: torch_geometric.data.Data object containing the graph
        probability: probability of masking each feature
    
    Returns:
        torch_geometric.data.Data: graph with masked node features
    """
    # 创建图的副本以避免修改原图
    masked_graph = graph.clone()
    
    # 对节点特征进行随机masking
    if hasattr(masked_graph, 'x') and masked_graph.x is not None:
        # 生成随机mask，True表示保留，False表示mask（设为0）
        mask = torch.rand(masked_graph.x.shape) > probability
        masked_graph.x = masked_graph.x * mask.float()
    
    return masked_graph

def _feature_noise(graph: torch_geometric.data.Data, noise_level=0.1):
    """
    Randomly adds noise to the features of the nodes in the graph.
    
    Args:
        graph: torch_geometric.data.Data object containing the graph
        noise_level: standard deviation of the Gaussian noise to add
    
    Returns:
        torch_geometric.data.Data: graph with noisy node features
    """
    # 创建图的副本以避免修改原图
    noisy_graph = graph.clone()
    
    # 对节点特征添加高斯噪声
    if hasattr(noisy_graph, 'x') and noisy_graph.x is not None:
        # 生成与特征矩阵相同形状的高斯噪声
        noise = torch.randn_like(noisy_graph.x) * noise_level
        noisy_graph.x = noisy_graph.x + noise
    
    return noisy_graph

def _feature_shuffling(graph: torch_geometric.data.Data):
    """
    Randomly shuffles the features of the nodes in the graph.
    
    Args:
        graph: torch_geometric.data.Data object containing the graph
    
    Returns:
        torch_geometric.data.Data: graph with shuffled node features
    """
    # 创建图的副本以避免修改原图
    shuffled_graph = graph.clone()
    
    # 对节点特征进行随机打乱
    if hasattr(shuffled_graph, 'x') and shuffled_graph.x is not None:
        # 生成随机排列索引并应用到节点特征
        perm_idx = torch.randperm(shuffled_graph.x.size(0))
        shuffled_graph.x = shuffled_graph.x[perm_idx]
    
    return shuffled_graph