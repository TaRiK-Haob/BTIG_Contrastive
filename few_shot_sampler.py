import json
import random
import os

####################################################################################
# 从cicids原始数据中采样，分为两个数据集-> 预训练数据集和微调数据集                  #
# 1.预训练：8000正常+4*500恶意                                                      #
# 2.微调：8000正常+1*2000恶意 -> few-shot微调 -> 测试集=(8000+1*2000)-2*N           #
####################################################################################

def get_few_shot_samples(file_path, num_samples_per_class=500, benign_samples=8000):
    """
    从CICIDS2017数据集中进行few-shot采样
    将DoS攻击合并为一类
    
    Args:
        file_path: 数据文件路径
        num_samples_per_class: 每个恶意类别的样本数量 (默认500)
        benign_samples: 良性样本数量 (默认8000)
    """
    
    # 定义标签映射 - DoS攻击合并为一类
    dos_labels = [3, 4, 5, 6]  # DoS相关攻击，将合并为一个类别
    other_attack_labels = [1, 2, 13, 14]  # FTP-Patator, SSH-Patator, PortScan, DDoS LOIT
    benign_label = 0
    
    # 存储不同类别的样本
    samples_by_label = {label: [] for label in dos_labels + other_attack_labels + [benign_label]}
    
    # 读取数据文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                label = data['label']
                if label in samples_by_label:
                    samples_by_label[label].append(data)
    
    # 将DoS攻击样本合并
    dos_samples = []
    for dos_label in dos_labels:
        dos_samples.extend(samples_by_label[dos_label])
    
    # 创建合并后的攻击类别列表：DoS(合并后) + 其他4个攻击类别
    attack_categories = {
        'dos': dos_samples,  # DoS攻击合并后的样本
        'ftp_patator': samples_by_label[1],  # FTP-Patator
        'ssh_patator': samples_by_label[2],  # SSH-Patator  
        'portscan': samples_by_label[13],    # PortScan
        'ddos_loit': samples_by_label[14]    # DDoS LOIT
    }

    # 从5个攻击类别中随机选择4个用于预训练
    attack_names = list(attack_categories.keys())
    print(f"可用攻击类别: {attack_names}")
    finetune_attack_name = random.choice(attack_names)
    pretrain_attack_names = [name for name in attack_names if name not in finetune_attack_name]

    print(f"预训练攻击类别: {pretrain_attack_names}")
    print(f"微调攻击类别: {finetune_attack_name}")
    
    # 预训练数据集
    pretrain_samples = []
    
    # 为每个预训练攻击类别采样
    for attack_name in pretrain_attack_names:
        attack_samples = attack_categories[attack_name]
        for sample in attack_samples:
            sample['class'] = attack_name
        if len(attack_samples) >= num_samples_per_class:
            selected_samples = random.sample(attack_samples, num_samples_per_class)
        else:
            selected_samples = attack_samples
            print(f"警告: {attack_name} 类别只有 {len(selected_samples)} 个样本，少于要求的 {num_samples_per_class} 个")
        pretrain_samples.extend(selected_samples)
    
    # 优化良性样本分配：先打乱，然后切片分配
    benign_samples_list = samples_by_label[benign_label][:]
    random.shuffle(benign_samples_list)
    
    for sample in benign_samples_list:
        sample['class'] = 'benign'

    # 为预训练数据集分配良性样本
    if len(benign_samples_list) >= benign_samples:
        selected_benign = benign_samples_list[:benign_samples]
        remaining_benign = benign_samples_list[benign_samples:]
    else:
        selected_benign = benign_samples_list
        remaining_benign = []
        print(f"警告: 良性样本只有 {len(selected_benign)} 个，少于要求的 {benign_samples} 个")
    
    pretrain_samples.extend(selected_benign)
    
    # 微调数据集
    finetune_samples = []
    
    # 为微调数据集添加攻击样本
    finetune_attack_count = 500
    finetune_attack_samples = attack_categories[finetune_attack_name]

    for sample in finetune_attack_samples:
        sample['class'] = finetune_attack_name

    if len(finetune_attack_samples) >= finetune_attack_count:
        selected_finetune_attack = random.sample(finetune_attack_samples, finetune_attack_count)
    else:
        selected_finetune_attack = finetune_attack_samples
        print(f"警告: 微调攻击类别 {finetune_attack_name} 只有 {len(selected_finetune_attack)} 个样本，少于要求的 {finetune_attack_count} 个")
    finetune_samples.extend(selected_finetune_attack)
    
    # 为微调数据集添加良性样本 (使用剩余的良性样本)
    finetune_benign_count = 2000
    if len(remaining_benign) >= finetune_benign_count:
        selected_finetune_benign = remaining_benign[:finetune_benign_count]
    else:
        # 如果剩余样本不够，从所有良性样本中重新随机选择
        if len(benign_samples_list) >= finetune_benign_count:
            selected_finetune_benign = random.sample(benign_samples_list, finetune_benign_count)
        else:
            selected_finetune_benign = benign_samples_list
        print(f"警告: 微调良性样本不足，实际选择了 {len(selected_finetune_benign)} 个样本")
    
    finetune_samples.extend(selected_finetune_benign)
    
    # 打乱样本顺序
    random.shuffle(pretrain_samples)
    random.shuffle(finetune_samples)
    
    # 保存预训练数据集
    with open('pretrain_data.jsonl', 'w', encoding='utf-8') as f:
        for sample in pretrain_samples:
            f.write(json.dumps(sample) + '\n')
    
    # 保存微调数据集
    with open('finetune_data.jsonl', 'w', encoding='utf-8') as f:
        for sample in finetune_samples:
            f.write(json.dumps(sample) + '\n')
    
    # 统计信息
    print(f"\n数据集统计:")
    print(f"预训练数据集总样本数: {len(pretrain_samples)}")
    print(f"微调数据集总样本数: {len(finetune_samples)}")
    
    # 统计各攻击类别的原始样本数
    print(f"\n各攻击类别原始样本数:")
    for name, samples in attack_categories.items():
        print(f"{name}: {len(samples)} 个样本")
    
    # 统计各类别样本数
    def count_attack_category(samples):
        """统计样本中各攻击类别的数量"""
        counts = {'benign': 0, 'dos': 0, 'ftp_patator': 0, 'ssh_patator': 0, 'portscan': 0, 'ddos_loit': 0}
        for sample in samples:
            label = sample['label']
            if label == 0:
                counts['benign'] += 1
            elif label in [3, 4, 5, 6]:
                counts['dos'] += 1
            elif label == 1:
                counts['ftp_patator'] += 1
            elif label == 2:
                counts['ssh_patator'] += 1
            elif label == 13:
                counts['portscan'] += 1
            elif label == 14:
                counts['ddos_loit'] += 1
        return counts
    
    pretrain_counts = count_attack_category(pretrain_samples)
    finetune_counts = count_attack_category(finetune_samples)
    
    print(f"预训练数据集各类别样本数: {pretrain_counts}")
    print(f"微调数据集各类别样本数: {finetune_counts}")
    
    return pretrain_samples, finetune_samples

# 使用示例
if __name__ == "__main__":

    
    # 调用函数
    pretrain_data, finetune_data = get_few_shot_samples('/home/hjh/Desktop/BTIG_pycontrast/data/CIC-IDS-2017/CIC-IDS-2017.jsonl')

