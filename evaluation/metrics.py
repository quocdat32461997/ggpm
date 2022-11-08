import os
import warnings
import os.path as osp
import torch
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import MolWt
import moses
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data import download_url, extract_zip
from torchmdnet.models.model import load_model
from torch_geometric.nn.models import SchNet
from ggpm import *


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


class SchNetwork(SchNet):
    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    @staticmethod
    def from_qm9_pretrained(root, target: int):
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


class Metrics:
    MOL_WEIGHTS = [400, 3000]

    def __init__(self, homo_net, lumo_net, batch_size=512, ks=[50, 500], num_worker=2, device='cpu'):
        self.homo_net, self.lumo_net = None, None
        if homo_net and lumo_net:
            self.homo_net = homo_net
            self.lumo_net = lumo_net
            self.homo_net.eval()
            self.lumo_net.eval()
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
