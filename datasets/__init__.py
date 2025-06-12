import torch_geometric
import torch
import random
import os
from . import TIGDataset
from . import FGDataset
import logging
import hydra

logger = logging.getLogger(__name__)

def _split_data(dataset, config, refresh=False):
    generator = torch.Generator().manual_seed(random.randint(0, 99999))

    train_file = os.path.join('temp', f'train_{config.dataset.name}_{config.task.binary}.pt')
    test_file = os.path.join('temp', f'test_{config.dataset.name}_{config.task.binary}.pt')
    val_file = os.path.join('temp', f'val_{config.dataset.name}_{config.task.binary}.pt')

    # 检查是否存在预处理后的数据集
    if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file):
        if refresh:
            logging.info('Refreshing dataset...')
            os.remove(train_file)
            os.remove(test_file)
            os.remove(val_file)
        else:
            logging.info('Loading existing dataset...')
        train_dataset = torch.load(train_file)
        test_dataset = torch.load(test_file)
        val_dataset = torch.load(val_file)
        logging.info('Dataset loaded')

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1], generator=generator)  
    torch.save(train_dataset, train_file)
    torch.save(test_dataset, test_file)
    torch.save(val_dataset, val_file)
    logging.info('Dataset saved & Loaded')

    # 添加多进程加载
    num_workers = 8  # 根据CPU核心数调整
    prefetch_factor = 8  # 预加载批次数
    
    train_loader = torch_geometric.loader.DataLoader(train_dataset, 
                            batch_size=config.hyperparameters.batch_size, 
                            shuffle=True,
                            num_workers=num_workers,
                            prefetch_factor=prefetch_factor,
                            pin_memory=True)  # GPU训练时使用
    
    test_loader = torch_geometric.loader.DataLoader(test_dataset, 
                           batch_size=config.hyperparameters.batch_size, 
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=True)
    
    val_loader = torch_geometric.loader.DataLoader(val_dataset, 
                          batch_size=config.hyperparameters.batch_size, 
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True)

    return train_loader, test_loader, val_loader

def get_dataloader(config):
    if config.graph == 'TIG':
        dataset = TIGDataset.TIGDataset(config)
        return _split_data(dataset, config)
    
    if config.graph == 'FG':
        dataset = FGDataset.FGDataset(config)
        return _split_data(dataset, config)