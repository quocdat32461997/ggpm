import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.data.data import Data
from torch_scatter import scatter
import os
import random
import gc
import pickle

from ggpm.chemutils import get_leaves
from ggpm.mol_graph import MolGraph
from ggpm.chemutils import *


class MoleculeDataset(Dataset):

    def __init__(self, data, vocab, avocab, batch_size):
        safe_data = []
        for mol_s, homo, lumo in data:
            hmol = MolGraph(mol_s)
            ok = True
            for node, attr in hmol.mol_tree.nodes(data=True):
                smiles = attr['smiles']
                ok &= attr['label'] in vocab.vmap
                # print('label', mol_s, attr['label'], ok)
                for i, s in attr['inter_label']:
                    ok &= (smiles, s) in vocab.vmap
                #    print('inter_label', mol_s, i, s, ok)
            if ok:
                safe_data.append([mol_s, homo, lumo])

        print(f'After pruning {len(data)} -> {len(safe_data)}')
        self.batches = [safe_data[i: i + batch_size] for i in range(0, len(safe_data), batch_size)]
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return MolGraph.tensorize(self.batches[idx], self.vocab, self.avocab)


class MolEnumRootDataset(Dataset):

    def __init__(self, data, vocab, avocab):
        self.batches = data
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.batches[idx])
        leaves = get_leaves(mol)
        smiles_list = set([Chem.MolToSmiles(mol, rootedAtAtom=i, isomericSmiles=False) for i in leaves])
        smiles_list = sorted(list(smiles_list))  # To ensure reproducibility

        safe_list = []
        for s in smiles_list:
            hmol = MolGraph(s)
            ok = True
            for node, attr in hmol.mol_tree.nodes(data=True):
                if attr['label'] not in self.vocab.vmap:
                    ok = False
            if ok:
                safe_list.append(s)

        if len(safe_list) > 0:
            return MolGraph.tensorize(safe_list, self.vocab, self.avocab)
        else:
            return None


class MolPairDataset(Dataset):

    def __init__(self, data, vocab, avocab, batch_size):
        self.batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        x, y = zip(*self.batches[idx])
        x = MolGraph.tensorize(x, self.vocab, self.avocab)[:-1]  # no need of order for x
        y = MolGraph.tensorize(y, self.vocab, self.avocab)
        return x + y


class DataFolder(object):

    def __init__(self, data_folder, batch_size, shuffle=True):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data_files) * 1000

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                batches = pickle.load(f)

            if self.shuffle:
                random.shuffle(batches)  # shuffle data before batch
            for batch in batches:
                yield batch

            del batches
            gc.collect()

class QM9Dataset(Dataset):
    atoms = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Read mol
        smiles = self.data[index]
        mol = get_mol(smiles)
        print(smiles)

        N = mol.GetNumAtoms()

        conf = mol.GetConformer()
        pos = conf.GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)

        type_idx = []
        atomic_number = []
        aromatic = []
        sp = []
        sp2 = []
        sp3 = []
        num_hs = []
        for atom in mol.GetAtoms():
            type_idx.append(QM9Dataset.atoms[atom.GetSymbol()])
            atomic_number.append(atom.GetAtomicNum())
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            sp.append(1 if hybridization == HybridizationType.SP else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

        z = torch.tensor(atomic_number, dtype=torch.long)

        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [QM9Dataset.bonds[bond.GetBondType()]]
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type,
                              num_classes=len(QM9Dataset.bonds)).to(torch.float)

        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
        edge_attr = edge_attr[perm]

        row, col = edge_index
        hs = (z == 1).to(torch.float)
        num_hs = scatter(hs[row], col, dim_size=N).tolist()

        x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(QM9Dataset.atoms))
        x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                          dtype=torch.float).t().contiguous()
        x = torch.cat([x1.to(torch.float), x2], dim=-1)

        name = mol.GetProp('_Name')
        return Data(x=x, z=z, pos=pos, edge_index=edge_index, \
            edge_attr=edge_attr, name=name)