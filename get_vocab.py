import argparse
import pandas as pd
from collections import Counter
from rdkit import Chem
from multiprocessing import Pool
from ggpm import *

from ggpm import MolGraph

def process(data):
    vocab = set()
    for line, i in zip(data, range(len(data))):
        #try:
        # trim space
        s = line.strip("\r\n ")

        # skip smiles if containing *
        if '*' in s:
            continue

        # extract fragment vocabs
        hmol = MolGraph(s)
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add((smiles, s))
        #except Exception as e:
        #    print('Error at line {}: {}'.format(i, e))
    return vocab

def fragment_process(data):
    counter = Counter()
    for smiles, i in zip(data, range(len(data))):
        #try:
        # trim space
        smiles = smiles.strip("\r\n ")

        # skip smiles if containing *
        if '*' in smiles:
            continue

        mol = get_mol(smiles)
        fragments = find_fragments(mol)
        for fsmiles, _ in fragments:
            counter[fsmiles] += 1
        #except Exception as e:
        #    print('Error at lin {}: {}'.format(i, e))
    return counter

if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--output', type=str, default='vocab.txt')
    parser.add_argument('--min_frequency', type=int, default=100)
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()

    # read data
    if args.data.endswith('.csv'):
        data = pd.read_csv(args.data)

        # drop empty row
        data = data.dropna(how='all', subset=['SMILES'])
        # remove SMIELS duplicates
        data = data.drop_duplicates(subset=['SMILES'])

        # save as cleaned-data
        data = data.reset_index(drop=True)
        data.to_csv('/'.join(args.data.split('/')[:-1] + ['cleaned_data.csv']), index=False)

        data = list(data['SMILES'])
    elif args.data.endswith('.txt'):
        with open(args.data) as file:
            data = [line for line in file.read().split('\n')]
    else:
        raise TypeError('Data files must be in csv format.')

    # parse into batches for parallel programming
    batch_size = len(data) // args.ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    counter = Counter()
    if args.ncpu == 1:
        # iterative process
        # get fragments
        counter = fragment_process(batches[0])
        # get vocabs
        vocab = [(x, y) for x, y in process(batches[0])]
    else:
        # pool process
        pool = Pool(args.ncpu)

        # get fragments
        counter_list = pool.map(fragment_process, batches)
        for cc in counter_list:
            counter += cc
        # get vocabs
        vocab_list = pool.map(process, batches)
        vocab = [(x, y) for vocab in vocab_list for x, y in vocab]

    # get fragments satisfying min_frequency
    fragments = [fragment for fragment, cnt in counter.most_common() if cnt >= args.min_frequency]
    MolGraph.load_fragments(fragments)

    # unique set of vocab and fragments
    vocab = list(set(vocab))
    fragments = set(fragments)

    with open(args.output, 'w') as file:
        for x,y in sorted(vocab):
            cx = Chem.MolToSmiles(Chem.MolFromSmiles(x))
            file.write(' '.join([x, y, str(cx in fragments)]) + '\n')
