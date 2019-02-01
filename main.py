import os
#import matplotlib.pyplot as plt
import argparse
from utils import *
from models.lstm_classifier import LSTMClassifier
from models.gru_classifier import GRUClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-context', help='Use previous utterances in model',
                        action='store_true')
    return parser.parse_args()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()

    train = load_data('data/utterances.train')
    val = load_data('data/utterances.valid')

    if args.with_context:
        train['utterance_t'] = merge(train)

    classes = train['dialog_act'].unique()
    num_classes = len(classes)
    max_len = train['utterance_t'].apply(len).max()

    (X_train, y_train), (vocab, word_to_idx, idx_to_word) = process_data(train, max_len)
    (X_val, y_val), (_,_,_) = process_data(val, max_len)

    print('Importing embeddings...')
    embds_path = '/projekte/slu/share/GoogleNews-vectors-negative300.bin' 
    w2v = load_bin_vec(embds_path, word_to_idx)
    add_unknown_words(w2v, word_to_idx)
    W, word_idx = get_W(w2v)
    print('Embeddings imported.')

    print('Initializing model...')
    #model = GRUClassifier(len(vocab), num_classes, max_len,
                          #custom_weights=W)
    model = LSTMClassifier(num_classes, max_len, W)
    optimizer = RMSprop(lr=0.01, decay=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=3, batch_size=256)
    print('Model trained')
    #plt.plot(history.history['acc'])


if __name__ == '__main__':
    main()
