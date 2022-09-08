import numpy as np
import pandas as pd
from torch.utils.data import Dataset

train_df = pd.read_csv('./data/train.csv')
# print(train_df.shape)
test_df = pd.read_csv('./data/test.csv')
# print(test_df.shape)


train = pd.DataFrame()
train['text'] = train_df['Description']
train['label'] = train_df['Class Index']