import pandas as pd
import numpy as np
import argparse
import random
from collections import Counter
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-context', help='Use previous utterances in model',
                        action='store_true')
    return parser.parse_args()

def load_weights(word_to_idx, source='glove'):
    emb_dir = '/projekte/slu/share/'
    if source == 'google':
        embds_path = emb_dir + 'GoogleNews-vectors-negative300.bin'
        weights = load_bin_vec(embds_path, word_to_idx)
    else:
        weights = {}
        embds_path = emb_dir + 'emb/glove.twitter.27B.50d.txt'
        with open(embds_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.split()
                word = line[0]
                if word in word_to_idx:
                    weights[word] = np.array(line[1:], dtype='float32')

    num_dims = weights[random.choice(list(weights.keys()))].shape[0]
    add_unknown_words(weights, word_to_idx, k=num_dims, min_df=0)
    return get_W(weights, k=num_dims)


def preprocess(series):
    """Preprocesses every row in a `Series`"""
    def delimit(sent):
        """Inserts special symbols for start and end into sentence"""
        arr = sent.split()
        if arr:
            arr.insert(0, '<START>')
            arr.append('<END>')
        else:
            arr = ['<EMPTY>']
        return arr
    return series.apply(delimit)


def process_seq(seq, mapping):
    """Gets the idx for every element in a sentence."""
    return [mapping[w] for w in seq]


def get_vocab(sents, top_words=None):
    """
    Gets the entire vocabulary of a list of tokenized sentences,
    then maps both every word to a specific index and every index
    to a specific word, so that each word type is represented as a
    single integer. Can be limited to a specific number of words. If
    `top_words` is None, returns all words.

    Args:
        sents (list): a list-like object of lists of tokens
        top_words (int): how many words to return, default None

    Returns:
        vocab (set): a set of all words types in the sentences
        word_to_idx (dict): a mapping of all words to an index
        idx_to_word (dict): a reverse mapping of all inidices to words
    """
    # We have to count the words for `top_words` scenario
    c = Counter()
    for sent in sents:
        c.update(sent)
        
    if top_words is not None:
        if top_words > len(c):
            raise Exception('top_words must be less than the number of words: {%d}' % len(c))
        vocab = set([w[0] for w in c.most_common(top_words)])
    else:
        vocab = set(c.keys())
        
    word_to_idx = {}
    idx_to_word = {}
                
    for idx, word in enumerate(c.keys()):
        word_to_idx[word] = idx
        idx_to_word[idx] = word
        
    return vocab, word_to_idx, idx_to_word

def load_data(path, train=True):
    """Loads data from `path` into a DataFrame."""
    COLUMNS = ['utterance_ID', 'dialog_act', 'utterance_t-3', 
           'utterance_t-2', 'utterance_t-1', 'utterance_t']

    if not train:
        COLUMNS.remove('dialog_act')
    
    df = (pd.read_csv(path, sep='\t|;', engine='python', names=COLUMNS)
            .set_index('utterance_ID')
            .astype(str))
    df[COLUMNS[2:]] = df[COLUMNS[2:]].apply(preprocess)
    return df


def process_data(data, max_len, top_words=None, custom_vocab=None):
    """
    Prepares the data for the model by calling `process_seq` on each sentence
    in the data, then padding the sequence to max_len. If `top_words` is None,
    then the data will include all words; otherwise, it returns only `top_words`
    words.

    Args:
        data (pd.DataFrame): all utterances
        max_len (int): the number of words in the longest utterance
        top_words (int): the n most common words to give to network

    Returns:
        X (pd.Series): the processed utterances 
        y (pd.Series): the processed intentions
        vocab, word_to_idx, idx_to_word: see `get_vocab`
    """
    # We have to calculate this every time even though we
    # only need it for the training data
    if custom_vocab is None:
        vocab, word_to_idx, idx_to_word = get_vocab(data['utterance_t'], top_words)

    label_to_idx = {label:idx for idx, label in enumerate(data['dialog_act'].unique())}

    X = sequence.pad_sequences(data['utterance_t'].apply(process_seq, args=[word_to_idx]),
                               maxlen=max_len)
    y = data['dialog_act'].map(label_to_idx).values
    y = to_categorical(y, len(data['dialog_act'].unique()))
    
    return (X, y), (vocab, word_to_idx, idx_to_word)


def merge(df):
    """Merges the context and the utterance into one column."""
    return (df['utterance_t-3'] + df['utterance_t-2'] + df['utterance_t-1'] \
            + df['utterance_t'])


# THE FOLLOWING TWO FUNCTIONS ARE TAKEN FROM A
# LECUTRE ON USING KERAS AT STUTTGART UNIVERSITY
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        # print(vocab_size)
        for line in range(vocab_size):
            # print(line)
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            # print(word)
            if word in vocab:
                # print(word)
                word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)

