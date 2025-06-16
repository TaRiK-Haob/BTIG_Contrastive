from sklearn.manifold import TSNE
from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models import get_model

def visualize(dataloader, config):
    model = get_model(config)

    model.load_state_dict(config.output_settings.initial_model_path)
    model.to(config.device)

    # TODO: 可视化代码


def tsne(dataloader):
    pass



def umap(dataloader,config):
    pass