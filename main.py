import hydra
from models import get_model
from datasets import get_dataloader
from trainer import train_contrastive
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    model = get_model(cfg)
    
    train_loader, val_loader, test_loader = get_dataloader(cfg)

    for batch_idx, (data1, data2) in enumerate(train_loader):
        print(data1.x)

if __name__ == "__main__":
    main()