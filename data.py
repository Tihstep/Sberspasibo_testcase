import random
import torch
from torch.utils.data import DataLoader, Dataset

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


