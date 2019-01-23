import pandas as pd

def preprocess(series):
    def delimit(sent):
        arr = sent.split()
        if arr:
            arr.insert(0, '<START>')
            arr.append('<END>')
        else:
            arr = ['<EMPTY>']
        return arr
    return series.apply(delimit)

def process_seq(seq, mapping):
    return [mapping[w] for w in seq]

def get_vocab(sents, top_words=None):
    c = Counter()
    for sent in sents:
        c.update(sent)
        
    if top_words is not None:
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
    COLUMNS = ['utterance_ID', 'dialog_act', 'utterance_t-3', 
           'utterance_t-2', 'utterance_t-1', 'utterance_t']
    
    df = pd.read_csv(path, sep='\t|;', engine='python', names=COLUMNS,
                     dtype=str).set_index('utterance_ID')
    df[COLUMNS[2:]] = df[COLUMNS[2:]].apply(preprocess)
    return df

def process_data(data, max_len, top_words=None):
    vocab, word_to_idx, idx_to_word = get_vocab(data['utterance_t'], top_words)
    label_to_idx = {label:idx for idx, label in enumerate(data['dialog_act'].unique())}
    
    X = sequence.pad_sequences(data['utterance_t'].apply(process_seq, args=[word_to_idx]),
                               maxlen=max_len)
    y = data['dialog_act'].map(label_to_idx).values
    
    return (X, y), (vocab, word_to_idx, idx_to_word)

