import os
from utils import *
from models.lstm_classifier import LSTMClassifier

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    train = load_data('data/utterances.train')
    val = load_data('data/utterances.valid')

    classes = train['dialog_act'].unique()
    num_classes = len(classes)
    max_len = train['utterance_t'].apply(len).max()

    (X_train, y_train), (vocab, word_to_idx, idx_to_word) = process_data(train, max_len)
    (X_val, y_val), (_,_,_) = process_data(val, max_len)

    model = LSTMClassifier(len(vocab), num_classes)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=3, batch_size=64)

if __name__ == '__main__':
    main()
