from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding

class LSTMClassifier(Sequential):
    def __init__(self, vocab_len, num_classes, embedding_dim,
                     dropout=0.2, lstm_layers=100):
        super(LSTMClassifier, self).__init__()
        #self.model = Sequential()
        self.add(Embedding(vocab_len, embedding_dim, input_length=embedding_dim))
        self.add(LSTM(lstm_layers, dropout=dropout, recurrent_dropout=dropout))
        self.add(Dense(num_classes, activation='softmax'))
        self.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

