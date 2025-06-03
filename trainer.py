import torch
import time
from torch.utils.tensorboard import SummaryWriter
import hydra
import logging

logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, config):
    device = torch.device(config.hyperparameters.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    epochs = config.hyperparameters.epochs
    lr = config.hyperparameters.learning_rate  # 学习率

    patience = config.hyperparameters.patience  # 容忍轮数
    min_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    decay_rate = config.hyperparameters.weight_decay
    decay_step = len(train_loader.dataset) * 2 // config.hyperparameters.batch_size + 1

    criterion = torch.nn.CrossEntropyLoss()

    if config.hyperparameters.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)

    # writer = SummaryWriter(log_dir=f'runs')

    # 训练时间统计
    total_train_start_time = time.time()
    total_fet_time = 0
    total_cct_time = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        epoch_fet_time = 0
        epoch_cct_time = 0

        for batch in train_loader:
            batch = batch.to(device)
            
            # FET时间统计（正向传播时间）
            fet_start_time = time.time()
            out = model(batch.x, batch.edge_index, batch.vector, batch.batch)
            fet_end_time = time.time()
            
            # CCT时间统计（反向传播时间）
            cct_start_time = time.time()
            loss = criterion(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cct_end_time = time.time()
            
            total_train_loss += loss.item()
            
            # 累计时间
            epoch_fet_time += (fet_end_time - fet_start_time)
            epoch_cct_time += (cct_end_time - cct_start_time)
        
        scheduler.step()
        total_fet_time += epoch_fet_time
        total_cct_time += epoch_cct_time
            
        # 验证阶段
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.vector, batch.batch)
                val_loss = criterion(out, batch.y)
                total_val_loss += val_loss.item()
        
        # 计算平均损失
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # writer.add_scalars('Loss', {'Train': avg_train_loss, 'Validation': avg_val_loss}, epoch)
        
        # 早停检查
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f'早停于epoch {epoch+1}')
            model.load_state_dict(best_model_state)
            break
        
        if logger:   
            logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, FET(正向传播): {epoch_fet_time:.4f}s, CCT(反向传播): {epoch_cct_time:.4f}s')
        else:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, FET(正向传播): {epoch_fet_time:.4f}s, CCT(反向传播): {epoch_cct_time:.4f}s')

    total_train_end_time = time.time()
    total_train_time = total_train_end_time - total_train_start_time

    # writer.close()
    
    # 记录训练时间统计
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