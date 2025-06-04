import hydra
from models import get_model
from datasets import get_dataloader
from trainer import Trainer
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    trainer = Trainer(cfg)
    
    mode = cfg.mode

    if mode == 'pretrain':
        logger.info("Starting contrastive pre-training...")
        trainer.train()
    elif mode == 'finetune':
        logger.info("Starting supervised fine-tuning...")
        pretrained_path = getattr(cfg, 'pretrained_model_path', None)
        trainer.finetune(pretrained_path)
    elif mode == 'both':
        logger.info("Running both pre-training and fine-tuning...")
        trainer.train()  # 先预训练
        # 使用刚刚保存的预训练模型进行微调
        trainer.finetune(cfg.output_settings.best_model_path)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'pretrain', 'finetune', or 'both'")

if __name__ == "__main__":
    main()