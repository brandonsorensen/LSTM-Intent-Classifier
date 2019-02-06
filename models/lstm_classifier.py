from models.base_classifier import BaseClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding

class LSTMClassifier(BaseClassifier):

    def __init__(self, num_classes, max_utter_len, weights,
                     dropout=0.2, lstm_layers=128, from_file=None):
        if from_file is None:
            self.model = _init_model(self, num_classes, max_utter_len,
                    weights, dropout=0.2, lstm_layers=128)
        else:
            self.model = load_model(from_file)
        self._set_methods()

    def _init_model(self, num_classes, max_utter_len, weights, dropout, lstm_layers):
        self.model = Sequential()
        self.model.add(Embedding(weights.shape[0], weights.shape[1], input_length=max_utter_len,
                           weights=[weights], mask_zero=False))
        self.model.add(LSTM(lstm_layers, dropout=dropout, return_sequences=True,
                 kernel_initializer='random_uniform', recurrent_initializer='glorot_uniform'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(num_classes, activation='softmax'))


