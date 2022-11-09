import os
import os.path as osp
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data import Data
from torch_scatter import scatter
from rdkit.Chem import AllChem
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
RDLogger.DisableLog('rdApp.*')

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}


class QM9(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = torch.load(path + '.pt')

    @staticmethod
    def process_data(smiles_list, root, path):
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        root = osp.expanduser(osp.normpath(root))
        makedirs(root)

        num_items = len(smiles_list)
        data_list = []
        for i, smiles in enumerate(tqdm(smiles_list)):
            try:

                mol = Chem.MolFromSmiles(smiles)
                AllChem.EmbedMolecule(mol)
                AllChem.UFFOptimizeMolecule(mol)

                N = mol.GetNumAtoms()

                type_idx = []
                atomic_number = []
                aromatic = []
                sp = []
                sp2 = []
                sp3 = []
                num_hs = []
                pos = []
                for i, atom in enumerate(mol.GetAtoms()):
                    pos.append(mol.GetConformer().GetAtomPosition(i))

                    type_idx.append(types[atom.GetSymbol()])
                    atomic_number.append(atom.GetAtomicNum())
                    aromatic.append(1 if atom.GetIsAromatic() else 0)
                    hybridization = atom.GetHybridization()
                    sp.append(1 if hybridization == HybridizationType.SP else 0)
                    sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                    sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
                pos = torch.tensor(pos, dtype=torch.float)
                z = torch.tensor(atomic_number, dtype=torch.long)

                row, col, edge_type = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_type += 2 * [bonds[bond.GetBondType()]]

                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)
                edge_attr = F.one_hot(edge_type,
                                      num_classes=len(bonds)).to(torch.float)

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_type = edge_type[perm]
                edge_attr = edge_attr[perm]

                row, col = edge_index
                hs = (z == 1).to(torch.float)
                num_hs = scatter(hs[row], col, dim_size=N).tolist()

                x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
                x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                                  dtype=torch.float).t().contiguous()
                x = torch.cat([x1.to(torch.float), x2], dim=-1)

                data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                            edge_attr=edge_attr, y=None, name=None, idx=i)

                data_list.append(data)
            except:
                pass
        torch.save(data_list, root + '/' + path + '.pt')

        print("Initial number of items: {}; Number of items ater processing: {}".format(num_items, len(data_list)))
