import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, word2vec, filters=50, kernels_height=[3, 5, 7], dropout_rate=0.5):
        super(CNN, self).__init__()
        self.__name__ = 'CNN'
        self.in_channels = 1
        self.out_channels = filters
        self.out_features = 2  # pos or neg

        self.embedding = nn.Embedding.from_pretrained(word2vec, freeze=False)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.in_channels, self.out_channels, (k, 50))
            for k in kernels_height
        ])  # kernel: (height, 50), where 50 is the embedding size
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.out_channels *
                            len(kernels_height), self.out_features)

    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(1)
        # embedded: (batch_size, 1, max_length, 50)
        conv_results = [F.relu(conv(embedded)).squeeze(3)
                        for conv in self.convs]
        # conv_results: [(batch_size, out_channels, max_length - k + 1), ...]
        pooled_results = [F.max_pool1d(conv, conv.size(2)).squeeze(2)
                          for conv in conv_results]
        # pooled_results: [(batch_size, out_channels), ...]
        return self.fc(self.dropout(torch.cat(pooled_results, dim=1)))


class RNN(nn.Module):
    def __init__(self, word2vec, hidden_size=64, num_layers=2, dropout_rate=0.5):
        super(RNN, self).__init__()
        self.__name__ = 'RNN'
        self.out_features = 2

        self.embedding = nn.Embedding.from_pretrained(word2vec, freeze=False)
        self.rnn = nn.LSTM(
            input_size=50,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        self.fc = nn.Linear(
            hidden_size * 2, self.out_features)  # bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        # output: (batch_size, max_length, hidden_size * 2)
        return self.fc(output[:, -1, :])


class MLP(nn.Module):
    def __init__(self, word2vec, hidden_size=64, dropout_rate=0.6):
        super(MLP, self).__init__()
        self.__name__ = 'MLP'
        self.out_features = 2
        self.max_length = 50

        self.embedding = nn.Embedding.from_pretrained(word2vec, freeze=False)
        self.fc1 = nn.Linear(50 * self.max_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        embedded = self.embedding(x).view(x.size(0), -1)
        # embedded: (batch_size, max_length * 50)
        output = F.relu(self.fc1(embedded))
        return self.fc2(self.dropout(output))
