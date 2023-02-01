import pandas as pd
import numpy as np
from data import SampleGenerator
from gmf import GMFEngine

gmf_config = {
              'alias': 'gmf_factor',
              'num_epoch': 200,
              'batch_size': 1024,
              'sgd_lr': 1e-3,
              'sgd_momentum': 0.9,
              'num_users': 6040,
              'num_items': 3706,
              'hidden_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0
}

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
# Definite Engines
GMFengine = GMFEngine(gmf_config)
MLPengine= MLPEngine(mlp_config)
NeuMFengine = NeuMFEngine(neumf_config)
