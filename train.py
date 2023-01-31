import pandas as pd
import numpy as np
from data import SampleGenerator

#gmf_config = pass

#mlp_config = pass

#neumf_config = pass

# Load Data
data_dir = 'Archive/interactions.csv'
user_item_data = pd.read_csv(data_dir, names=['uid', 'mid', 'interaction'])
# Reindex
user_id = user_item_data[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
user_item_data = pd.merge(user_item_data, user_id, on=['uid'], how='left')
item_id = user_item_data[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
user_item_data = pd.merge(user_item_data, item_id, on=['mid'], how='left')
user_item_data = user_item_data[['userId', 'itemId', 'rating', 'timestamp']]
# DataLoader for training
sample_generator = SampleGenerator(interactions=user_item_data)
evaluate_data = sample_generator.evaluate_data
