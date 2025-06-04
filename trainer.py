import torch
import time
import hydra
import logging

logger = logging.getLogger(__name__)

def train_contrastive(model, train_loader, val_loader, config):
    logger.info("Start Training...")

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

        for batch_idx, (data1, data2) in enumerate(train_loader):
            model.train()
            data1 = data1.to(device)
            data2 = data2.to(device)
            total_train_loss = 0
            epoch_fet_time = 0
            epoch_cct_time = 0


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