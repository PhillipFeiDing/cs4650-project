import torch
import torch.nn as nn


def create_embedding_layer(embedding_matrix, trainable=False):
    num_embeddings, embedding_dim = embedding_matrix.size()
    emb_layer = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    emb_layer.load_state_dict({'weight': embedding_matrix})
    emb_layer.weight.requires_grad = trainable
    return emb_layer, embedding_dim


class LSTMClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, \
                 num_layers=1, bidirectional=True, \
                 pretrained_embeddings=None, trainable_embeddings=False, \
                 use_contextual_embeddings=False):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding, embedding_dim = create_embedding_layer(pretrained_embeddings, trainable_embeddings)
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
