from BTIG_GNN import GCN_multi_scale

def get_model(config):
    """
    根据配置文件获取模型实例
    :param config: 配置对象
    :return: 模型实例
    """
    if config.model.name == 'BTIG_GNN':
        model = GCN_multi_scale(config)
        return model