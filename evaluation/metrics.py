import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from ggpm import *


class Metrics:
    OPV_MOL_WEIGHTS = [400, 3000]

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

    def get_recon_metrics(self, output_set, test_set, train_set=None):
        import moses
        return moses.get_all_metrics(gen=output_set, k=self.ks, n_jobs=self.num_worker,
                                     device=self.device, batch_size=self.batch_size,
                                     test=test_set, train=train_set)

    def mol_weight_indicator(self, output_set, test_set, train_set=None):
        # mw_list = self.get_recon_n_sample_metrics(output_set, test_set, train_set)#['weight']

        mw_list = [MolWt(Chem.MolFromSmiles(smiles)) for smiles in output_set[:30]]
        valids = [1 if Metrics.OPV_MOL_WEIGHTS[0] <= mw <= Metrics.OPV_MOL_WEIGHTS[1] else 0 for mw in mw_list]
        valids = torch.tensor(valids)
        return {'mw_valids': valids, 'mean_mw_valids': valids.mean(), 'std_mw_valids': valids.std()}

    def property_indicator(self, output_smiles, root, use_processed):
        import evaluation.datasets as datasets
        from torch_geometric.loader import DataLoader

        dataset = datasets.QM9(root=root)
        if not use_processed:
            dataset.process(output_smiles)
            dataset = datasets.QM9(root=root)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)

        homos, lumos = [], []
        with torch.no_grad():
            for data in data_loader:
                homos.append(self.homo_net(data.z, data.pos, data.batch))
                lumos.append(self.lumo_net(data.z, data.pos, data.batch))

        homos = torch.stack(homos)
        lumos = torch.stack(lumos)

        # homos and lumos in torch.tensor
        valids = torch.ones(len(homos))

        # both negative
        valids *= (homos < 0) & (lumos < 0)

        # abs_lumos < abs_homos
        valids *= lumos.abs() < homos.abs()

        # lumos - homos > 0.8
        valids *= (lumos - homos) > 0.8

        # cast to float and compute mean
        valids = valids.float()

        return {'prop_valids': valids,
                'mean_prop_valids': valids.mean(),
                'std_prop_valids': valids.std(),
                'mean_homos': homos.mean(),
                'std_homos': homos.std(),
                'mean_lumos': lumos.mean(),
                'std_lumos': lumos.std()}

    def get_optim_metrics(self, output_smiles, root, test_set, train_set, use_processed=False):

        # get property-indiactor
        metrics = self.property_indicator(output_smiles[:30], root, use_processed)

        # get molecule-weight indicator
        metrics_ = self.mol_weight_indicator(
            output_set=output_smiles, test_set=test_set, train_set=train_set)

        # update metrics
        for k, v in metrics_.items():
            metrics[k] = v
        return metrics
