def train_contrastive(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (data1, data2) in enumerate(dataloader):
        data1 = data1.to(device)
        data2 = data2.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        loss, z1, z2 = model(data1, data2)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)