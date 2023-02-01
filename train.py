import pandas as pd
import numpy as np
from data import SampleGenerator
from gmf import GMFEngine
from mlp import MLPEngine
from nmf import NMFEngine

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

mlp_config = {
              'num_epoch': 200,
              'batch_size': 256,  # 1024,
              'sgd_lr': 1e-3,
              'sgd_momentum': 0.9,
              'num_users': 6040,
              'num_items': 3706,
              'hidden_dim': 8,
              'num_negative': 4,
              'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 7,
}

neumf_config = {
                'num_epoch': 200,
                'batch_size': 1024,
                'sgd_lr': 1e-3,
                'sgd_momentum': 0.9,
                'num_users': 6040,
                'num_items': 3706,
                'hidden_dim_mf': 8,
                'hidden_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.01,
                'use_cuda': True,
                'device_id': 7
}

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
NMFengine = NeuMF(neumf_config)

for epoch in range(neumf_config['num_epoch']):
    train_loader = sample_generator.instance_a_train_loader(neumf_config['num_negative'], neumf_config['batch_size'])
    NMFengine.train_an_epoch(train_loader, epoch_id=epoch)
    hit10 = NeuMFengine.evaluate(evaluate_data, epoch_id=epoch)
    NeuMFengine.save(neumf_config['alias'], epoch)
