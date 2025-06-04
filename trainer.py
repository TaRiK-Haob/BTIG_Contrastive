import torch
import torch.nn as nn
import time
import hydra
import logging
from models import get_model
from datasets import get_dataloader

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config, model = None, train_loader = None, val_loader = None):
        if model is not None:
            logger.info("Using provided model. Overriding the configuration model.")
            self.model = model
            self.model_name = model.__class__.__name__
        else:
            logger.info("No model provided. Initializing model from configuration...")
            self.model_name = config.model.name
            self.model = get_model(config)
            logger.info(f"Model {self.model_name} initialized successfully.")
            logger.info(f"Model configuration: {self.model}")

        self.config = config

        if train_loader is None or val_loader is None:
            logger.info("Train and validation loaders not provided. Loading dataset from configuration...")
            logger.info("Loading dataset...")
            self.train_loader, _, self.val_loader = get_dataloader(config)
            logger.info("Dataset loaded successfully.")
        else:
            logger.info("Using provided train and validation loaders. Overriding the configuration dataset.")
            self.train_loader = train_loader
            self.val_loader = val_loader

    def train(self):
        if self.model_name == 'TIG_CONTRASTIVE':
            logger.info("Using TIG_CONTRASTIVE model for training.")
            return train_contrastive(self.model, self.train_loader, self.val_loader, self.config)
        else:
            raise ValueError(f"Unsupported model: {self.config.model.name}. Please check your configuration.")

    def finetune(self, pretrained_model_path=None):
        """
        有监督微调方法
        """
        if self.model_name == 'TIG_CONTRASTIVE':
            logger.info("Starting supervised fine-tuning...")
            
            # 如果提供了预训练模型路径，加载预训练权重
            if pretrained_model_path:
                logger.info(f"Loading pretrained weights from {pretrained_model_path}")
                self.model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
            
            return finetune_supervised(self.model, self.train_loader, self.val_loader, self.config)
        else:
            raise ValueError(f"Unsupported model for fine-tuning: {self.model_name}")

def train_contrastive(model, train_loader, val_loader, config):
    logger.info("==================Start Training==================")

    min_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    device = torch.device(config.hyperparameters.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    lr = config.hyperparameters.learning_rate
    if config.hyperparameters.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练时间统计
    total_train_start_time = time.time()
    total_fet_time = 0
    total_cct_time = 0

    for epoch in range(config.hyperparameters.epochs):
        total_train_loss = 0
        epoch_fet_time = 0
        epoch_cct_time = 0

        for batch_idx, (data1, data2) in enumerate(train_loader):
            model.train()
            data1 = data1.to(device)
            data2 = data2.to(device)

            # 前向传播
            fet_start_time = time.time()
            loss, z1, z2 = model(data1, data2)
            fet_end_time = time.time()

            # 反向传播
            cct_start_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cct_end_time = time.time()

            total_train_loss += loss.item()

            epoch_fet_time += (fet_end_time - fet_start_time)
            epoch_cct_time += (cct_end_time - cct_start_time)

        total_fet_time += epoch_fet_time
        total_cct_time += epoch_cct_time

        # 验证阶段
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data1, data2) in enumerate(val_loader):
                data1 = data1.to(device)
                data2 = data2.to(device)

                loss, z1, z2 = model(data1, data2)

                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # 早停检查
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= config.hyperparameters.patience:
            logger.info(f'早停于epoch {epoch+1}')
            model.load_state_dict(best_model_state)
            break

        if logger:   
            logger.info(f'Epoch {epoch+1}/{config.hyperparameters.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, FET(正向传播): {epoch_fet_time:.4f}s, CCT(反向传播): {epoch_cct_time:.4f}s')
        else:
            print(f'Epoch {epoch+1}/{config.hyperparameters.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, FET(正向传播): {epoch_fet_time:.4f}s, CCT(反向传播): {epoch_cct_time:.4f}s')

    total_train_end_time = time.time()
    total_train_time = total_train_end_time - total_train_start_time

    if logger:
        logger.info(f"\n训练时间统计:")
        logger.info(f"总训练时间: {total_train_time:.4f}s")
        logger.info(f"总FET时间(正向传播): {total_fet_time:.4f}s")
        logger.info(f"总CCT时间(反向传播): {total_cct_time:.4f}s")
        logger.info(f"平均每epoch FET时间: {total_fet_time/(epoch+1):.4f}s")
        logger.info(f"平均每epoch CCT时间: {total_cct_time/(epoch+1):.4f}s")
    
    torch.save(best_model_state, f'{config.output_settings.best_model_path}')
    logger.info(f'最佳模型已保存: {config.output_settings.best_model_path}')

    return total_train_time, total_fet_time, total_cct_time


def finetune_supervised(model, train_loader, val_loader, config):
    """
    有监督微调函数
    """
    logger.info("==================Start Fine-tuning==================")
    
    device = torch.device(config.hyperparameters.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 微调学习率，通常比预训练小
    finetune_lr = getattr(config.hyperparameters, 'finetune_lr', config.hyperparameters.learning_rate * 0.1)
    
    # 选择性冻结编码器参数（可选）
    freeze_encoder = getattr(config.hyperparameters, 'freeze_encoder', False)
    if freeze_encoder:
        logger.info("Freezing encoder parameters...")
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    # 设置优化器，只优化需要梯度的参数
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if config.hyperparameters.optimizer == 'adam':
        optimizer = torch.optim.Adam(trainable_params, lr=finetune_lr)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 早停参数
    min_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    best_val_acc = 0.0
    
    finetune_epochs = getattr(config.hyperparameters, 'finetune_epochs', config.hyperparameters.epochs)
    
    # 训练时间统计
    total_train_start_time = time.time()
    total_fet_time = 0
    total_cct_time = 0
    
    for epoch in range(finetune_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        epoch_fet_time = 0
        epoch_cct_time = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):

            data = data.to(device)
            labels = data.y.to(device)

            # 前向传播
            fet_start_time = time.time()
            logits = model(data, mode='finetune')
            loss = criterion(logits, labels)
            fet_end_time = time.time()
            
            # 反向传播
            cct_start_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cct_end_time = time.time()
            
            total_train_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            epoch_fet_time += (fet_end_time - fet_start_time)
            epoch_cct_time += (cct_end_time - cct_start_time)
        
        total_fet_time += epoch_fet_time
        total_cct_time += epoch_cct_time
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_loader):

                data = data.to(device)
                labels = data.y.to(device)
                
                logits = model(data, mode='finetune')
                loss = criterion(logits, labels)
                
                total_val_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # 早停检查（基于验证准确率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            min_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= config.hyperparameters.patience:
            logger.info(f'早停于epoch {epoch+1}')
            model.load_state_dict(best_model_state)
            break
        
        logger.info(f'Epoch {epoch+1}/{finetune_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, FET: {epoch_fet_time:.4f}s, CCT: {epoch_cct_time:.4f}s')
    
    total_train_end_time = time.time()
    total_train_time = total_train_end_time - total_train_start_time
    
    logger.info(f"\n微调时间统计:")
    logger.info(f"总微调时间: {total_train_time:.4f}s")
    logger.info(f"总FET时间(正向传播): {total_fet_time:.4f}s")
    logger.info(f"总CCT时间(反向传播): {total_cct_time:.4f}s")
    logger.info(f"平均每epoch FET时间: {total_fet_time/(epoch+1):.4f}s")
    logger.info(f"平均每epoch CCT时间: {total_cct_time/(epoch+1):.4f}s")
    
    # 保存微调后的模型
    torch.save(best_model_state, f'{config.output_settings.best_finetune_model}')
    logger.info(f'最佳模型已保存: {config.output_settings.best_finetune_model}')

    # finetune_model_path = getattr(config.output_settings, 'finetune_model_path', 'best_finetune_model.pth')
    # torch.save(best_model_state, finetune_model_path)
    # logger.info(f'最佳微调模型已保存: {finetune_model_path}')
    logger.info(f'最佳验证准确率: {best_val_acc:.2f}%')
    
    return total_train_time, total_fet_time, total_cct_time, best_val_acc