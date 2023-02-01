import torch
from engine import Engine


class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.hidden_dim = config['hidden_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.hidden_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.hidden_dim)

        self.Linear = torch.nn.Linear(in_features=self.hidden_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.Linear(element_product)
        interaction = self.logistic(logits)
        return interaction


class GMFEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        self.model = GMF(config)
        if config['use_cuda'] is True:
            assert torch.cuda.is_available(), 'CUDA is not available'
            torch.cuda.set_device(config['device_id'])
            self.model.cuda()
        super(GMFEngine, self).__init__(config)
