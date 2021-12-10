import math
import torch
import torch.nn as nn


def create_embedding_layer(embedding_matrix):
    num_embeddings, embedding_dim = embedding_matrix.size()
    emb_layer = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    emb_layer.load_state_dict({'weight': embedding_matrix})
    return emb_layer, embedding_dim


class LSTMClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, \
                 num_layers=1, bidirectional=True, \
                 pretrained_embeddings=None, \
                 use_contextual_embeddings=False):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding, embedding_dim = create_embedding_layer(pretrained_embeddings)
        elif use_contextual_embeddings:
            self.embedding = None
        else:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)
        self.mlp = nn.Linear(
            in_features=hidden_dim * (2 if bidirectional else 1),
            out_features=1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        embeddings = self.embedding(x) if self.embedding is not None else x
        output, _ = self.rnn(embeddings)
        return self.output(self.mlp(output[:, -1, :])).view(-1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = torch.swapaxes(x, 0, 1)
        x = x + self.pe[:x.size(0)]
        return torch.swapaxes(self.dropout(x), 0, 1)

class TransformerClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers=1, nhead=8, dim_feedforward=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = nn.Linear(
            in_features=embedding_dim,
            out_features=1)
        self.output = nn.Sigmoid()
    
    def forward(self, x):
        embeddings = self.pos_encoder(self.embedding(x) * math.sqrt(self.embedding_dim))
        return self.output(self.mlp(self.transformer(embeddings).mean(axis=1))).view(-1)


class RobertaClassificationModel(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.output = nn.Sigmoid()
    
    def forward(self, x):
        return self.output(self.bert(x).logits).view(-1)