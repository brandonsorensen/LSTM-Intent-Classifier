import os
import argparse
from utils import *
from models.lstm_classifier import LSTMClassifier
from models.gru_classifier import GRUClassifier

def add_test_vocab(test_data, total_vocab, word_to_idx, idx_to_word):
    test_vocab = [word for row in test_data['utterance_t']
            for word in row if word not in total_vocab]
    for word in test_vocab:
        idx = len(word_to_idx)
        word2idx[word] = idx
        idx2word[idx] = word
        vocab.add(word)

def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()

    train = load_data('data/utterances.train')
    val = load_data('data/utterances.valid')

    train = pd.concat([train, val]).sample(frac=1) # Merge train & val then shuffle
    val_split = len(val) / len(train)

    if args.with_context:
        train['utterance_t'] = merge(train)

    classes = train['dialog_act'].unique()
    num_classes = len(classes)
    max_len = train['utterance_t'].apply(len).max()

    (X_train, y_train), (vocab, word_to_idx, idx_to_word) = process_data(train, max_len)
    add_test_vocab(load_data('data/utterances.test', train=False),
        vocab, word_to_idx, idx_to_word)

    print('Importing embeddings...')
    load_weights(word_to_idx, source='glove')
    print('Embeddings imported.')

    print('Initializing model...')
    model = LSTMClassifier(num_classes, max_len, W)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    history = model.fit(X_train, y_train, validation_split=val_split,
                        epochs=3, batch_size=256)
    print('Model trained')
    model_name = 'lstm_model.h5'
    print('Saving model to ./' + model_name)
    model.save(model_name)

