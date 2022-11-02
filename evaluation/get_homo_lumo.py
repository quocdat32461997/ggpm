import argparse
import pandas as pd

from ggpm.evaluation.utils import MoleculeStats

def calculate_homo_lumo(args):

    # read data
    smiles_list = pd.read_csv(args.path)['SMILES']

    mol_stats = MoleculeStats(memory_limit=args.memory_limit, num_threads=args.num_threads)

    homos, lumos = mol_stats.get_homo_lumo(smiles_list)
    
    # save homos and lumos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--memory-limit', default=4)
    parser.add_argument('--num-threads', default=4)

    args = parser.parse_args()

    calculate_homo_lumo(args)