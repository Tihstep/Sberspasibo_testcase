import torch


class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        pass
    def forward(self, user_indices, item_indices):
        pass

    def init_weight(self):
        pass


class GMFEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        pass
