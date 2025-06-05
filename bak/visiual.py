import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from datasets import get_dataloader

logger = logging.getLogger(__name__)

class TSNEVisualizer:
    def __init__(self, config, model_path, device='cuda'):
        """
        t-SNE可视化器
        
        Args:
            config_path: 配置文件路径
            model_path: 预训练模型路径
            device: 设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        self.config = config
        
        # 初始化模型
        self.model = get_model(self.config)
        
        # 加载预训练权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            logger.info(f"模型权重加载完成: {model_path}")
        else:
            logger.warning(f"模型文件不存在: {model_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # 加载数据
        self.train_loader, self.test_loader, self.val_loader = get_dataloader(self.config)
        logger.info("数据加载完成")
        
        # 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")
        
    def extract_embeddings(self, data_loader, max_samples=1000, split_name="validation"):
        """
        提取编码器输出的嵌入向量，按标签均匀采样
        
        Args:
            data_loader: 数据加载器
            max_samples: 最大采样数量
            split_name: 数据集名称（用于日志）
            
        Returns:
            embeddings: 嵌入向量 [N, embedding_dim]
            labels: 对应的标签 [N]
        """
        logger.info(f"从{split_name}数据集提取嵌入向量...")
        
        # 第一轮：收集所有数据和标签
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                # 处理不同的数据格式
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, _ = batch_data
                else:
                    data = batch_data
                    
                data = data.to(self.device)
                
                # 获取编码器输出
                try:
                    # 方法1：使用encode方法
                    if hasattr(self.model, 'encode'):
                        embedding = self.model.encode(data.x, data.edge_index, data.batch)
                    # 方法2：使用encoder.forward
                    elif hasattr(self.model, 'encoder'):
                        embedding = self.model.encoder(data.x, data.edge_index, data.batch, return_embedding=True)
                    # 方法3：使用finetune模式
                    else:
                        embedding = self.model(data, mode='finetune')
                        if isinstance(embedding, torch.Tensor) and embedding.dim() == 2 and embedding.size(1) == self.model.num_classes:
                            # 如果返回的是logits，我们需要使用编码器输出
                            embedding = self.model.encode(data.x, data.edge_index, data.batch)
                            
                except Exception as e:
                    logger.warning(f"获取嵌入向量失败，尝试备用方案: {e}")
                    try:
                        # 备用方案：直接调用模型并获取中间表示
                        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'encoder'):
                            # 对于TIG_CONTRASTIVE模型
                            embedding = self.model.encoder.encoder(data.x, data.edge_index, data.batch, return_embedding=True)
                        else:
                            # 最后的备用方案
                            logits = self.model(data, mode='finetune')
                            # 如果logits是分类输出，我们无法使用它进行可视化
                            if logits.size(1) == self.model.num_classes:
                                logger.error("无法获取合适的嵌入向量用于可视化")
                                continue
                            embedding = logits
                    except Exception as e2:
                        logger.error(f"所有获取嵌入向量的方法都失败了: {e2}")
                        continue
            
            # 确保embedding是正确的格式
            if isinstance(embedding, torch.Tensor):
                all_embeddings.append(embedding.cpu().numpy())
                all_labels.append(data.y.cpu().numpy())
            else:
                logger.warning(f"跳过无效的嵌入向量，类型: {type(embedding)}")
                continue
            
            if batch_idx % 10 == 0:
                total_samples = sum(len(labels) for labels in all_labels)
                logger.info(f"已处理 {total_samples} 个样本...")
    
        if not all_embeddings:
            logger.error("未能提取到任何嵌入向量")
            return np.array([]), np.array([])
        
        # 合并所有数据
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.concatenate(all_labels)
        
        logger.info(f"总共提取了 {len(all_embeddings)} 个嵌入向量")
        
        # 按标签进行均匀采样
        embeddings, labels = self._balanced_sampling(all_embeddings, all_labels, max_samples)
        
        logger.info(f"均匀采样完成: {len(embeddings)} 个嵌入向量，形状 {embeddings.shape}")
        return embeddings, labels

    def _balanced_sampling(self, embeddings, labels, max_samples):
        """
        按标签进行均匀采样
        
        Args:
            embeddings: 所有嵌入向量
            labels: 所有标签
            max_samples: 最大采样数量
            
        Returns:
            sampled_embeddings: 采样后的嵌入向量
            sampled_labels: 采样后的标签
        """
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        num_classes = len(unique_labels)
        
        logger.info(f"类别分布: {dict(zip(unique_labels, label_counts))}")
        
        # 计算每个类别应该采样的数量
        samples_per_class = max_samples // num_classes
        remaining_samples = max_samples % num_classes
        
        logger.info(f"每个类别采样 {samples_per_class} 个样本，剩余 {remaining_samples} 个样本")
        
        sampled_embeddings = []
        sampled_labels = []
        
        # 为每个类别进行采样
        for i, label in enumerate(unique_labels):
            label_mask = labels == label
            label_embeddings = embeddings[label_mask]
            label_indices = np.where(label_mask)[0]
            
            # 当前类别的样本数量
            current_class_samples = samples_per_class
            if i < remaining_samples:  # 将剩余样本分配给前几个类别
                current_class_samples += 1
            
            # 如果当前类别样本数少于需要采样的数量，则全部使用
            available_samples = len(label_embeddings)
            actual_samples = min(current_class_samples, available_samples)
            
            if available_samples > 0:
                # 随机采样
                if actual_samples < available_samples:
                    selected_indices = np.random.choice(
                        available_samples, 
                        size=actual_samples, 
                        replace=False
                    )
                else:
                    selected_indices = np.arange(available_samples)
                
                sampled_embeddings.append(label_embeddings[selected_indices])
                sampled_labels.append(np.full(actual_samples, label))
                
                logger.info(f"类别 {label}: 可用样本 {available_samples}, 采样 {actual_samples}")
            else:
                logger.warning(f"类别 {label}: 没有可用样本")
        
        if not sampled_embeddings:
            logger.error("均匀采样失败：没有采样到任何样本")
            return np.array([]), np.array([])
        
        # 合并采样结果
        sampled_embeddings = np.vstack(sampled_embeddings)
        sampled_labels = np.concatenate(sampled_labels)
        
        # 随机打乱顺序
        shuffle_indices = np.random.permutation(len(sampled_embeddings))
        sampled_embeddings = sampled_embeddings[shuffle_indices]
        sampled_labels = sampled_labels[shuffle_indices]
        
        # 显示最终的类别分布
        final_unique_labels, final_counts = np.unique(sampled_labels, return_counts=True)
        logger.info(f"采样后类别分布: {dict(zip(final_unique_labels, final_counts))}")
        
        return sampled_embeddings, sampled_labels
    
    def plot_tsne(self, embeddings, labels, perplexity=30, n_iter=1000, 
                  save_path=None, title="t-SNE Visualization", figsize=(12, 8)):
        """
        使用t-SNE进行降维可视化
        
        Args:
            embeddings: 嵌入向量
            labels: 标签
            perplexity: t-SNE参数
            n_iter: 迭代次数
            save_path: 保存路径
            title: 图标题
            figsize: 图大小
        """
        logger.info("执行t-SNE降维...")
        
        # 检查样本数量
        n_samples = len(embeddings)
        logger.info(f"样本数量: {n_samples}")
        
        # 动态调整perplexity - 必须小于样本数量
        # 建议perplexity在5-50之间，且小于n_samples
        max_perplexity = min(50, n_samples - 1)
        adjusted_perplexity = min(perplexity, max_perplexity)
        
        # 确保perplexity至少为5（如果样本数足够）
        if n_samples > 10:
            adjusted_perplexity = max(5, adjusted_perplexity)
        else:
            adjusted_perplexity = max(2, min(adjusted_perplexity, n_samples // 2))
        
        logger.info(f"原始perplexity: {perplexity}, 调整后perplexity: {adjusted_perplexity}")
        
        # 如果样本数量太少，给出警告
        if n_samples < 10:
            logger.warning(f"样本数量过少 ({n_samples})，t-SNE可视化效果可能不理想")
        
        # 标准化特征
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # t-SNE降维
        tsne = TSNE(n_components=2, perplexity=adjusted_perplexity, n_iter=n_iter, 
                   random_state=42, verbose=1)
        embeddings_2d = tsne.fit_transform(embeddings_scaled)
        
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 获取唯一标签和颜色
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        # 绘制散点图
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=f'Class {label}', 
                       alpha=0.7, s=20, edgecolors='black', linewidth=0.1)
        
        # 设置图形属性
        plt.title(f"{title}\n(Samples: {n_samples}, Perplexity: {adjusted_perplexity})", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        
        # 图例设置
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                  frameon=True, fancybox=True, shadow=True)
        
        # 网格和样式
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"t-SNE可视化已保存到: {save_path}")
        
        plt.show()
        
        return embeddings_2d
    
    def plot_class_statistics(self, embeddings, labels, save_path=None):
        """
        绘制类别统计信息
        
        Args:
            embeddings: 嵌入向量
            labels: 标签
            save_path: 保存路径
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 子图1：柱状图
        bars = ax1.bar(unique_labels, counts, alpha=0.7, 
                       color=plt.cm.tab10(np.linspace(0, 1, len(unique_labels))))
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class Label', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 子图2：饼图
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        wedges, texts, autotexts = ax2.pie(counts, labels=[f'Class {label}' for label in unique_labels], 
                                           autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        # 美化饼图文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 添加统计信息
        total_samples = len(labels)
        stats_text = f"""统计信息:
总样本数: {total_samples}
类别数量: {len(unique_labels)}
平均每类: {total_samples/len(unique_labels):.1f}
最大类别: {max(counts)} (Class {unique_labels[np.argmax(counts)]})
最小类别: {min(counts)} (Class {unique_labels[np.argmin(counts)]})
平衡度: {min(counts)/max(counts):.3f}"""
        
        fig.text(0.02, 0.5, stats_text, fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                 verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"类别统计图已保存到: {save_path}")
        
        plt.show()
    
    def plot_sampling_comparison(self, original_labels, sampled_labels, save_path=None):
        """
        对比采样前后的类别分布
        
        Args:
            original_labels: 原始标签
            sampled_labels: 采样后的标签
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 原始分布
        unique_orig, counts_orig = np.unique(original_labels, return_counts=True)
        bars1 = ax1.bar(unique_orig, counts_orig, alpha=0.7, 
                        color=plt.cm.Set3(np.linspace(0, 1, len(unique_orig))))
        
        for bar, count in zip(bars1, counts_orig):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_orig)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax1.set_title(f'原始分布 (总计: {len(original_labels)})', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Class Label')
        ax1.set_ylabel('Sample Count')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 采样后分布
        unique_samp, counts_samp = np.unique(sampled_labels, return_counts=True)
        bars2 = ax2.bar(unique_samp, counts_samp, alpha=0.7, 
                        color=plt.cm.Set2(np.linspace(0, 1, len(unique_samp))))
        
        for bar, count in zip(bars2, counts_samp):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_samp)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax2.set_title(f'均匀采样后 (总计: {len(sampled_labels)})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Class Label')
        ax2.set_ylabel('Sample Count')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"采样对比图已保存到: {save_path}")
        
        plt.show()
    
    def visualize_all_splits(self, max_samples=1000, perplexity=30, output_dir="visualizations"):
        """
        对所有数据集分割进行可视化
        
        Args:
            max_samples: 每个分割的最大样本数
            perplexity: t-SNE参数
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 验证集可视化
        if self.val_loader:
            try:
                logger.info("可视化验证集...")
                val_embeddings, val_labels = self.extract_embeddings(
                    self.val_loader, max_samples, "validation")
                
                # 检查是否有足够的样本进行可视化
                if len(val_embeddings) > 3:  # 至少需要4个样本
                    # t-SNE可视化
                    self.plot_tsne(val_embeddings, val_labels, perplexity=perplexity,
                                  save_path=os.path.join(output_dir, "validation_tsne.png"),
                                  title="t-SNE Visualization - Validation Set (Balanced Sampling)")
                    
                    # 类别统计
                    self.plot_class_statistics(val_embeddings, val_labels,
                                             save_path=os.path.join(output_dir, "validation_class_stats.png"))
                else:
                    logger.warning(f"验证集样本数量过少 ({len(val_embeddings)})，跳过可视化")
                    
            except Exception as e:
                logger.error(f"验证集可视化失败: {e}")
        
        # 测试集可视化
        if self.test_loader:
            try:
                logger.info("可视化测试集...")
                test_embeddings, test_labels = self.extract_embeddings(
                    self.test_loader, max_samples, "test")
                
                # 检查是否有足够的样本进行可视化
                if len(test_embeddings) > 3:  # 至少需要4个样本
                    # t-SNE可视化
                    self.plot_tsne(test_embeddings, test_labels, perplexity=perplexity,
                                  save_path=os.path.join(output_dir, "test_tsne.png"),
                                  title="t-SNE Visualization - Test Set (Balanced Sampling)")
                    
                    # 类别统计
                    self.plot_class_statistics(test_embeddings, test_labels,
                                             save_path=os.path.join(output_dir, "test_class_stats.png"))
                else:
                    logger.warning(f"测试集样本数量过少 ({len(test_embeddings)})，跳过可视化")
                    
            except Exception as e:
                logger.error(f"测试集可视化失败: {e}")
        
        # 训练集可视化（可选）
        if self.train_loader:
            try:
                logger.info("可视化训练集...")
                train_embeddings, train_labels = self.extract_embeddings(
                    self.train_loader, max_samples, "training")
                
                if len(train_embeddings) > 3:
                    # t-SNE可视化
                    self.plot_tsne(train_embeddings, train_labels, perplexity=perplexity,
                                  save_path=os.path.join(output_dir, "training_tsne.png"),
                                  title="t-SNE Visualization - Training Set (Balanced Sampling)")
                    
                    # 类别统计
                    self.plot_class_statistics(train_embeddings, train_labels,
                                             save_path=os.path.join(output_dir, "training_class_stats.png"))
                else:
                    logger.warning(f"训练集样本数量过少 ({len(train_embeddings)})，跳过可视化")
                
            except Exception as e:
                logger.error(f"训练集可视化失败: {e}")
        
        logger.info(f"所有可视化完成，结果保存在: {output_dir}")

def visualize_with_config(config_path, model_path, output_dir="visualizations", 
                         max_samples=1000, perplexity=30, device='cuda'):
    """
    便捷函数：使用配置文件和模型路径进行可视化
    
    Args:
        config_path: 配置文件路径
        model_path: 模型文件路径
        output_dir: 输出目录
        max_samples: 最大样本数
        perplexity: t-SNE参数
        device: 设备
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建可视化器
    visualizer = TSNEVisualizer(config_path, model_path, device)
    
    # 执行可视化
    visualizer.visualize_all_splits(max_samples, perplexity, output_dir)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    主函数：使用Hydra加载配置并执行可视化
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 获取模型路径
    model_path = cfg.output_settings.best_model_path
    if not os.path.exists(model_path):
        # 尝试使用微调模型
        model_path = cfg.output_settings.best_finetune_model
        if not os.path.exists(model_path):
            logger.error(f"找不到模型文件: {model_path}")
            return
    
    # 创建可视化器
    visualizer = TSNEVisualizer(
        config=cfg,
        model_path=model_path,
        device=cfg.hyperparameters.device
    )
    
    # 执行可视化
    output_dir = f"visualizations/{cfg.model.name}"
    visualizer.visualize_all_splits(
        max_samples=1000,
        perplexity=30,
        output_dir=output_dir
    )


if __name__ == "__main__":
    # 方式1: 使用Hydra（推荐）
    main()
    
    # 方式2: 直接调用（可选）
    # visualize_with_config(
    #     config_path="../config/config.yaml",
    #     model_path="../outputs/2024-01-01/12-00-00/best_pretrain_model.pth",
    #     output_dir="../visualizations/manual_run"
    # )