import torch
from torch.utils.data import Dataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
PADDING_VALUE = 0
UNK_VALUE = 1


def split_train_val_test(df, props=[.8, .1, .1], shuffle=False):
    assert round(sum(props), 2) == 1 and len(props) >= 2
    total = len(df)
    if shuffle:
        df = df.sample(frac=1)
    bound1, bound2 = int(props[0] * total), int((props[0] + props[1]) * total)
    train_df = df.iloc[:bound1]
    val_df = df.iloc[bound1:bound2]
    test_df = df.iloc[bound2:]
    
    return train_df, val_df, test_df


def generate_vocab_map(df, cutoff=2):
    vocab = {"": PADDING_VALUE, "UNK": UNK_VALUE}
    
    freq = {}
    for sentence in df["tokenized"]:
        for word in sentence:
            freq[word] = 1 if word not in freq else freq[word] + 1
    this_id = 0
    reversed_vocab = {PADDING_VALUE: "", UNK_VALUE: "UNK"}
    for word, freq in freq.items():
        if freq > cutoff:
            while this_id == PADDING_VALUE or this_id == UNK_VALUE:
                this_id += 1
            vocab[word] = this_id
            reversed_vocab[this_id] = word
            this_id += 1
    
    return vocab, reversed_vocab


class HeadlineDataset(Dataset):

    def __init__(self, vocab, df, max_length=50, use_elmo=False):
        self.vocab = vocab
        self.df = df
        self.max_length = max_length
        self.use_elmo = use_elmo
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: int):
        frame = self.df.iloc[index]
        curr_label = frame["label"]
        sentence = frame["tokenized"][:self.max_length]
        if not self.use_elmo:
            sentence = frame["tokenized"][:self.max_length]
            unk_id = self.vocab["UNK"]
            word_tensor = [self.vocab.get(word, unk_id) for word in sentence]
            tokenized_word_tensor = torch.LongTensor(word_tensor)
            return tokenized_word_tensor, curr_label
        else:
            return sentence, curr_label


def collate_fn(batch, padding_value=PADDING_VALUE):
    inputs, y_labels = zip(*batch)
    y_labels = torch.FloatTensor(y_labels)
    if type(inputs[0]) is torch.Tensor:
        padded_tokens = pad_sequence(inputs, batch_first=True, padding_value=padding_value)
        return padded_tokens, y_labels
    else:
        return inputs, y_labels