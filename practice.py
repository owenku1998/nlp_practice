import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

tokenizer = get_tokenizer("basic_english")


counter = Counter()
for i in range(0, 119999):
    counter.update(tokenizer(train_df.iat[i, 2]))

vocab = Vocab(counter)
vocab_size = len(vocab)
print(f"Vocab size of {vocab_size}")
