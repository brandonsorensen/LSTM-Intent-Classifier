import pandas as pd
from collections import Counter
from keras.preprocessing import sequence


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
            raise Exception(f'top_words must be less than the number of words: {len(c)}')
        vocab = set([w[0] for w in c.most_common(top_words)])
    else:
        vocab = set(c.keys())
        
    word_to_idx = {}
    idx_to_word = {}
                
    for idx, word in enumerate(c.keys()):
        word_to_idx[word] = idx
        idx_to_word[idx] = word
        
    return vocab, word_to_idx, idx_to_word

def load_data(path):
    """Loads data from `path` into a DataFrame."""
    COLUMNS = ['utterance_ID', 'dialog_act', 'utterance_t-3', 
           'utterance_t-2', 'utterance_t-1', 'utterance_t']
    
    df = pd.read_csv(path, sep='\t|;', engine='python', names=COLUMNS,
                     dtype=str).set_index('utterance_ID')
    df[COLUMNS[2:]] = df[COLUMNS[2:]].apply(preprocess)
    return df

def process_data(data, max_len, top_words=None):
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
    vocab, word_to_idx, idx_to_word = get_vocab(data['utterance_t'], top_words)
    label_to_idx = {label:idx for idx, label in enumerate(data['dialog_act'].unique())}

    X = sequence.pad_sequences(data['utterance_t'].apply(process_seq, args=[word_to_idx]),
                               maxlen=max_len)
    y = data['dialog_act'].map(label_to_idx).values
    
    return (X, y), (vocab, word_to_idx, idx_to_word)

def merge(df):
    """Merges the context and the utterance into one column."""
    return (df['utterance_t-3'] + df['utterance_t-2'] + df['utterance_t-1'] \
            + df['utterance_t'])

