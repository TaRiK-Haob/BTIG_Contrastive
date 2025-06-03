import hydra
from models import get_model

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    model = get_model(cfg)



if __name__ == "__main__":
    main()