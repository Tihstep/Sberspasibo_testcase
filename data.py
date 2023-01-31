import random
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

random.seed(0)


class UserItemDataDataset(Dataset):
    """Wrapper, convert <user, item, data> Tensor into Pytorch Dataset

        args:
            target_tensor: torch.Tensor, the corresponding action for <user, item> pair
    """
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[0][index], self.data_tensor[1][index], self.data_tensor[2][index]

    def __len__(self):
        return self.data_tensor.size()


class SampleGenerator(object):
    """Construct dataset for NCF

            args:
                data: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
    """
    def __init__(self, interactions):
        assert 'userId' in interactions.columns
        assert 'itemId' in interactions.columns
        assert 'data' in interactions.columns

        self.interactions = interactions
        self.user_pool = set(self.interactions['userId'].unique())
        self.item_pool = set(self.interactions['itemId'].unique())
        # create negative item samples for NCF learning
        #self.negatives = self._sample_negative(interactions)
        #self.train_interactions, self.test_interactions = self._split_loo(self.interactions)

    @staticmethod
    def _split_loo(interactions):
        pass

    def _sample_negative(self, interactions):
        pass
