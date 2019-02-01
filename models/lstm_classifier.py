from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding

class LSTMClassifier(Sequential):
    def __init__(self, num_classes, max_utter_len, weights,
                     dropout=0.2, lstm_layers=128):
        super(LSTMClassifier, self).__init__()
        self.add(Embedding(weights.shape[0], weights.shape[1], input_length=max_utter_len,
                           weights=[weights], mask_zero=False))
        self.add(LSTM(lstm_layers, dropout=dropout, return_sequences=True,
                 kernel_initializer='random_uniform', recurrent_initializer='glorot_uniform'))
        self.add(GlobalMaxPooling1D())
        self.add(Dense(num_classes, activation='softmax'))

