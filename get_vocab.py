import sys
import argparse
import pandas as pd
from rdkit import Chem
from multiprocessing import Pool

from ggpm import MolGraph

def process(data):
    vocab = set()
    for line, i in zip(data, range(len(data))):
        try:
            #if i == 2:
            #    continue
            s = line.strip("\r\n ")
            hmol = MolGraph(s)
            for node,attr in hmol.mol_tree.nodes(data=True):
                smiles = attr['smiles']
                vocab.add( attr['label'] )
                for i,s in attr['inter_label']:
                    vocab.add((smiles, s))
        except Exception as e:
            print("Error at molecule {}".format(i))
    return vocab

if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--output', type=str, default='vocab.txt')
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()

    # read data
    if args.data.endswith('.csv'):
        data = pd.read_csv(args.data)
    else:
        raise TypeError('Data files must be in csv format.')

    # drop empty row
    data = data.dropna(how='all', subset=['SMILES'])
    # remove SMIELS duplicates
    data = data.drop_duplicates(subset=['SMILES'])

    # save as cleaned-data
    data = data.reset_index(drop=True)
    data.to_csv('/'.join(args.data.split('/')[:-1] + ['cleaned_data.csv']))

    # parase into batches for parallel programming
    data = list(data['SMILES'])
    batch_size = len(data) // args.ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(args.ncpu)
    vocab_list = pool.map(process, batches)
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
    vocab = list(set(vocab))

    with open('/'.join(args.data.split('/')[:-1] + [args.output]), 'w') as file:
        for x,y in sorted(vocab):
            file.write(' '.join([x, y]) + '\n')
