import random
import torch
import pandas as pd
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
                data: pd.DataFrame, which contains 3 columns = ['userId', 'itemId', 'interactions']
    """
    def __init__(self, interactions):
        assert 'userId' in interactions.columns
        assert 'itemId' in interactions.columns
        assert 'data' in interactions.columns

        self.interactions = interactions
        self.user_pool = set(self.interactions['userId'].unique())
        self.item_pool = set(self.interactions['itemId'].unique())
        # create negative item samples for NCF learning
        self.negatives = self._sample_negative(interactions)
        self.train_interactions, self.test_interactions = self._split_loo(self.interactions)

    @staticmethod
    def _split_loo(interactions):
        """Split dataset by 0.8/0.2 train/test proportions"""
        train, test = train_test_split(interactions, test_size=0.2)
        return train['userId', 'itemId', 'data'], test['userId', 'itemId', 'data']

    def _sample_negative(self, ratings):
        """Return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def instance_a_train_loader(self, num_negatives, batch_size):
        """Instance train loader for one training epoch"""
        users, items, interactions = [], [], []
        train_interactions = pd.merge(self.train_interactions, self.negatives[['userId', 'negative_items']], on='userId')
        train_interactions['negatives'] = train_interactions['negative_items'].apply(lambda x: random.sample(x, num_negatives))
        for row in train_interactions.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            interactions.append(float(row.interactions))
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                interactions.append(float(0))  # negative samples get 0 rating
        data_tensor = torch.cat([torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(interactions)])
        dataset = UserItemDataDataset(data_tensor=data_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self):
        """Create evaluate data"""
        test_interactions = pd.merge(self.test_interactions, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_interactions.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]