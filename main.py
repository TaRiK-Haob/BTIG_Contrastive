import hydra
from models import get_model
from datasets import get_dataloader
from trainer import train_model
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    model = get_model(cfg)
    
    train_loader, val_loader, test_loader = get_dataloader(cfg)

    train_model(model, train_loader, val_loader, cfg)




if __name__ == "__main__":
    main()