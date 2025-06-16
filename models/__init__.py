from . import TIG_CONTRASTIVE
from . import TESTMODEL

def get_model(config):
    """
    根据配置文件获取模型实例
    :param config: 配置对象
    :return: 模型实例
    """
    if config.model.name == 'TIG_CONTRASTIVE':
        return TIG_CONTRASTIVE.TIG_CONTRASTIVE(config)
    
    if config.model.name == 'TESTMODEL':
        return TESTMODEL.TESTMODEL(config)