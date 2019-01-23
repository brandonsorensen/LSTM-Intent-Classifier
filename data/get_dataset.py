import pandas as pd

def main():
    df = pd.read_csv('utterances.train', sep='\t|;')
    print(df.head())

if __name__ == '__main__':
    main()
