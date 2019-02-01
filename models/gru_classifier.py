from keras.models import Sequential
from keras.layers import Dense, GRU, Bidirectional, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding

class GRUClassifier(Sequential):
    def __init__(self, vocab_len, num_classes, embedding_dim,
                 custom_weights, dropout=0.5):
        super(GRUClassifier, self).__init__()
        self.add(Embedding(custom_weights.shape[0], custom_weights.shape[1], input_length=embedding_dim,
                           weights=[custom_weights]))
        self.add(Bidirectional(GRU(64)))
        self.add(Dense(64, activation = 'relu'))
        self.add(Dropout(dropout))
        self.add(Dense(64, activation = 'relu'))
        self.add(Dropout(dropout))
        self.add(BatchNormalization())
        self.add(Dense(num_classes, activation = 'softmax'))
        self.compile(loss='sparse_categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])

