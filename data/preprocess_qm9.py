# preprocess_qm9.py
import os
import argparse
from functools import partial
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

def read_and_extract(files, path_dir):
    # Function to read and extract SMILES and properties from QM9.
    files = files if isinstance(files, list) else [files]

    res = []
    for file in tqdm(files):

        # read file
        with open(os.path.join(path_dir, file)) as f:
            file = f.read().split('\n')
        #print(file)

        # get smiles
        index = 3 + int(file[0])
        smiles = [file[index].split('\t')[0]]
        # skip if CNO
        if smiles[-1] in ['C', 'N', 'O']:
            continue
        res.append(smiles)

        # get HOMO and LUMO properties
        res[-1].extend([float(x) for x in file[1].split(' ')[-1].split('\t')[6:8]])
        #print(res[-1])
    return res


def main(args):
    # Pipeline to parse and extract all SMILES and properties in QM9 dataset.
    # read data
    data = os.listdir(args.data)

    # parse into batches for parallel programming
    batch_size = len(data) // args.ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    # pool process
    pool = Pool(args.ncpu)

    # create abstract function
    _read_and_extract = partial(read_and_extract, path_dir=args.data)
    
    # get SMILES and properties
    _data = pool.map(_read_and_extract, batches)
    data = list(_data)
    data = np.array([x for x in _data])[0]

    # save data
    data = pd.DataFrame(data=data,
                        columns=['SMILES', 'HOMO', 'LUMO'])
    data.to_csv(args.data + '.csv', index=False)
    pass


if __name__ == '__main__':
    # initialize parser
    parser = argparse.ArgumentParser()

    # get path to data dir
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ncpu', type=int, default=1)

    main(parser.parse_args())