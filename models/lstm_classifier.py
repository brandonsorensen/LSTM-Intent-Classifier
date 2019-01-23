from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding

class LSTMClassifier(object):
    def __init__(self, vocab_len, num_classes, embedding_dim=None,
                     dropout=0.2, lstm_layers=100, **kwargs):
        super(LSTMClassifier, self).__init__(**kwargs)
        if embedding_dim is None:
            self.embedding_dim = vocab_len
        else:
            self.embedding_dim = embedding_dim

        self.model = Sequential()
        self.model.add(Embedding(vocab_len, embedding_dim, input_length=embedding_dim))
        self.model.add(LSTM(lstm_layers, dropout=dropout, recurrent_dropout=dropout))
        self.model.add(Dense(num_classes, activation='softmax'))

