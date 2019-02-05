from keras.models import load_model
from utils import *

def main():
    test = load_data('data/utterances.test', train=False)
    X_test = sequence.pad_sequences(test['utterance_t'].apply(process_seq, args=[word_idx]),
                                maxlen=106)
    model = load_model('lstm_model.h5')
    predictions = model.predict_classes(X_test)
    with open('predictions.txt', 'w') as f:
        for pred in predictions:
            f.write(str(pred) + '\n')

if __name__ == '__main__':
    main()
