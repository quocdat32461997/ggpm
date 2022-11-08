import os
import warnings
import os.path as osp
import torch
from rdkit.Chem import MolFromSmiles, AllChem
from rdkit.Chem.Descriptors import MolWt
import moses
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data import download_url, extract_zip, Data
from torch_geometric.data import Data, InMemoryDataset
from torchmdnet.models.model import load_model
from torch_geometric.nn.models import SchNet
from torch_scatter import scatter
from ggpm import *
from tqdm import tqdm
import torch.nn.functional as F
from typing import Callable, List, Optional

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

qm9_target_dict = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}


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


class QM9(InMemoryDataset):
    def __init__(self, data: List[str], root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, _ = torch.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def processed_file_names(self) -> str:
        return 'data_v3.pt'

    def download(self):
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self, smiles_list):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

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

                name = None  # mol.GetProp('_Name')

                data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                            edge_attr=edge_attr, y=None, name=name, idx=i)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
            except:
                pass
        torch.save(self.collate(data_list), self.processed_paths[0])

        print("Initial number of items: {}; Number of items ater processing: {}".format(num_items, len(data_list)))


class SchNetwork(SchNet):
    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    @staticmethod
    def from_qm9_pretrained(root, dataset, target: int):
        import ase
        import schnetpack as psk

        assert target >= 0 and target <= 12
        is_dipole = target == 0

        units = [1] * 12
        units[0] = ase.units.Debye
        units[1] = ase.units.Bohr**3
        units[5] = ase.units.Bohr**2

        root = osp.expanduser(osp.normpath(root))
        makedirs(root)
        folder = 'trained_schnet_models'
        if not osp.exists(osp.join(root, folder)):
            path = download_url(SchNet.url, root)
            extract_zip(path, root)
            os.unlink(path)

        name = f'qm9_{qm9_target_dict[target]}'
        path = osp.join(root, 'trained_schnet_models', name, 'best_model')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            state = torch.load(path, map_location='cpu')

        net = SchNet(hidden_channels=128, num_filters=128, num_interactions=6,
                     num_gaussians=50, cutoff=10.0, dipole=is_dipole,
                     atomref=None)

        net.embedding.weight = state.representation.embedding.weight

        for int1, int2 in zip(state.representation.interactions,
                              net.interactions):
            int2.mlp[0].weight = int1.filter_network[0].weight
            int2.mlp[0].bias = int1.filter_network[0].bias
            int2.mlp[2].weight = int1.filter_network[1].weight
            int2.mlp[2].bias = int1.filter_network[1].bias
            int2.lin.weight = int1.dense.weight
            int2.lin.bias = int1.dense.bias

            int2.conv.lin1.weight = int1.cfconv.in2f.weight
            int2.conv.lin2.weight = int1.cfconv.f2out.weight
            int2.conv.lin2.bias = int1.cfconv.f2out.bias

        net.lin1.weight = state.output_modules[0].out_net[1].out_net[0].weight
        net.lin1.bias = state.output_modules[0].out_net[1].out_net[0].bias
        net.lin2.weight = state.output_modules[0].out_net[1].out_net[1].weight
        net.lin2.bias = state.output_modules[0].out_net[1].out_net[1].bias

        mean = state.output_modules[0].atom_pool.average
        net.readout = 'mean' if mean is True else 'add'

        dipole = state.output_modules[0].__class__.__name__ == 'DipoleMoment'
        net.dipole = dipole

        net.mean = state.output_modules[0].standardize.mean.item()
        net.std = state.output_modules[0].standardize.stddev.item()

        if state.output_modules[0].atomref is not None:
            net.atomref.weight = state.output_modules[0].atomref.weight
        else:
            net.atomref = None

        net.scale = 1. / units[target]

        return net


class PropertyPredictor:
    MODEL_VERSION = "ANI1-equivariant_transformer/epoch=359-val_loss=0.0004-test_loss=0.0120.ckpt"

    def __init__(self):
        self.model = load_model(PropertyPredictor.MODEL_VERSION, derivative=True)
        self.model.eval()

    def __call__(self, mol_list):
        positions = []
        atomic_nums = []
        batch_idxs = []
        for batch_idx, mol in enumerate(mol_list):
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
            mol.GetConformer()

            for i, atom in enumerate(mol.GetAtoms()):
                pos = mol.GetConformer().GetAtomPosition(i)
                positions.append([pos.x, pos.y, pos.z])
                atomic_nums.append(atom.GetAtomicNum())

            batch_idxs.extend([batch_idx] * len(mol.Getatoms()))

        atomic_nums = torch.tensor(atomic_nums, dtype=torch.long)
        positions = torch.tensor(positions)
        batch_idxs = torch.tensor(batch_idxs, dtype=torch.long)

        return atomic_nums, positions, batch_idxs


class Metrics:
    MOL_WEIGHTS = [400, 3000]

    def __init__(self, property_predictor=None, tokenizer=None, batch_size=512, ks=[50, 500], num_worker=2, device='cpu'):
        if property_predictor and tokenizer:
            self.property_predictor = property_predictor
            self.property_predictor.eval()
            self.tokenizer = tokenizer
        self.ks = ks
        self.num_worker = num_worker
        self.device = device
        self.batch_size = batch_size

    def get_recon_n_sample_metrics(self, output_set, test_set, train_set=None):
        return moses.get_all_metrics(gen=output_set, k=self.ks, n_jobs=self.num_worker,
                                     device=self.device, batch_size=self.batch_size,
                                     test=test_set, train=train_set)

    def mol_weight_indicator(self, output_set, test_set, train_set=None):
        # mw_list = self.get_recon_n_sample_metrics(output_set, test_set, train_set)#['weight']

        mw_list = [MolWt(MolFromSmiles(smiles)) for smiles in output_set[:30]]
        valids = [1 if Metrics.MOL_WEIGHTS[0] <= mw <= Metrics.MOL_WEIGHTS[1] else 0 for mw in mw_list]

        return valids, sum(valids) / len(valids)

    def property_indicator(self, output_smiles):
        # get homos and lumos
        output_smiles = self.tokenizer(output_smiles, return_tensors='pt',
                                       add_special_tokens=True, padding=True)['input_ids']
        with torch.no_grad():
            #homos, lumos = self.property_predictor.predict(output_smiles)
            homos = self.property_predictor.predict(output_smiles)
        # homos and lumos in torch.tensor
        #valids = torch.ones(len(homos))

        # both negative
        #valids *= (homos < 0) & (lumos < 0)

        # abs_lumos < abs_homos
        #valids *= lumos.abs() < homos.abs()

        # lumos - homos > 0.8
        #valids *= (lumos - homos) > 0.8

        # cast to float and compute mean
        #valids = valids.float()
        # return valids, valids.float().mean()
        return homos, None

    def get_optimization_metrics(self, output_smiles, test_set, train_set):
        prop_valids, property_i = None, None

        # get property-indiactor
        if self.property_predictor:
            prop_valids, property_i = self.property_indicator(output_smiles[:30])

        # get molecule-weight indicator
        mw_valids, mol_weight_i = self.mol_weight_indicator(
            output_set=output_smiles, test_set=test_set, train_set=train_set)

        return {'prop_valids': prop_valids, 'property_i': property_i,
                'mw_valids': mw_valids, 'mol_weight_i': mol_weight_i}
