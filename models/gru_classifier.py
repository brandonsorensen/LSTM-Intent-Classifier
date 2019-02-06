from models.base_classifier import BaseClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense, GRU, Bidirectional, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding

class GRUClassifier(BaseClassifier):
    def __init__(self, num_classes, max_len,
                 weights, dropout=0.2, from_file=None):
        if from_file is None:
            self._init_model(num_classes, max_len, weights, dropout)
        else:
            self.model = load_model(from_file)
        self._set_methods()

    def _init_model(self, num_classes, max_len, weights, dropout):
        self.model = Sequential()
        self.model.add(Embedding(weights.shape[0], weights.shape[1], input_length=max_len,
                           weights=[weights]))
        self.model.add(Bidirectional(GRU(64)))
        self.model.add(Dense(64, activation = 'relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(64, activation = 'relu'))
        self.model.add(Dropout(dropout))
        self.model.add(BatchNormalization())
        self.model.add(Dense(num_classes, activation = 'softmax'))


def load_gru(path):
    return GRUClassifier(None, None, None, None, None, from_file=path)
