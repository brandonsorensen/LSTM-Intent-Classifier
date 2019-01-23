from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

class LSTMClassifier(Sequential):
    def __init__(self, vocab_len, num_classes, embedding_dim=None,
                     dropout=0.2, lstm_layers=100):
        if embedding_dim is None:
            self.embedding_dim = vocab_len
        else:
            self.embedding_dim = embedding_dim

        self.add(Embedding(vocab_len), embedding_dim, input_length=embedding_dim))
        self.add(LSTM(lstm_layers, dropout=dropout, recurrent_dropout=dropout))
        self.add(Dense(num_classes, activation='softmax'))
        self.compile(loss='sparse_categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])

