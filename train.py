import pandas as pd
import numpy as np
from data import SampleGenerator

#gmf_config = pass

#mlp_config = pass

#neumf_config = pass

# Load Data
data_dir = 'Archive/interactions.csv'
user_item_data = pd.read_csv(data_dir, names=['uid', 'mid', 'interaction'])
