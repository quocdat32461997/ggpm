import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt


class Metrics:
    OPV_MOL_WEIGHTS = [400, 3000]

    def __init__(self, homo_net, lumo_net, batch_size=512, ks=[50, 10000], num_worker=2, device='cpu'):
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
        metrics =  moses.get_all_metrics(gen=output_set, k=self.ks, n_jobs=self.num_worker,
                                     device=self.device, batch_size=self.batch_size,
                                     test=test_set, train=train_set)
        # reconstruction accuracy
        print(output_set[0], test_set[0])
        acc = torch.tensor([x == y for x, y in zip(output_set, test_set)], dtype=torch.float).mean()
        metrics['recon_acc'] = acc.item()

        return metrics
        

    def get_mol_weight_indicator(self, output_set, test_set, train_set=None):
        mw_list = [MolWt(Chem.MolFromSmiles(smiles)) for smiles in output_set]
        valids = [1 if Metrics.OPV_MOL_WEIGHTS[0] <= mw <= Metrics.OPV_MOL_WEIGHTS[1] else 0 for mw in mw_list]
        valids = torch.tensor(valids, dtype=torch.float)
        valid_idxs = torch.nonzero(valids).reshape(-1).tolist()

        return {'mw_valid_idxs': valid_idxs, 'mw_valids': valids, 'mean_mw_valids': valids.mean(), 'std_mw_valids': valids.std()}

    def get_property_indicator(self, output_smiles, root, path, use_processed):
        from evaluation.datasets import QM9
        from torch_geometric.loader import DataLoader

        processed_path = path
        if not use_processed:
            QM9.process_data(output_smiles, root, path)
            processed_path = root + '/' + path
        dataset = QM9(processed_path)
        data_loader = DataLoader(dataset.data, batch_size=self.batch_size)

        homos, lumos = [], []
        with torch.no_grad():
            for data in data_loader:
                homos.append(self.homo_net(data.z, data.pos, data.batch))
                lumos.append(self.lumo_net(data.z, data.pos, data.batch))

        homos = torch.concat(homos).view(-1)
        lumos = torch.concat(lumos).view(-1)

        # homos and lumos in torch.tensor
        valids = torch.ones(len(homos))

        # both negative
        valids *= (homos < 0) & (lumos < 0)

        # abs_lumos < abs_homos
        valids *= lumos.abs() < homos.abs()

        # lumos - homos > 0.8
        valids *= (lumos - homos) >= 0.8

        # cast to float and compute mean
        valids = valids.float()

        return {'prop_valids': valids,
                'mean_prop_valids': valids.mean(),
                'std_prop_valids': valids.std(),
                'mean_homos': homos.mean(),
                'std_homos': homos.std(),
                'mean_lumos': lumos.mean(),
                'std_lumos': lumos.std(),
                'prop_valid_idxs': [d.idx for d in dataset.data]}

    def get_optim_metrics(self, root, path, output_smiles, test_smiles, train_smiles, use_processed=False):

        # get property-indiactor
        metrics = self.get_property_indicator(output_smiles, root, path, use_processed)

        metrics_ = self.get_mol_weight_indicator(
            output_set=output_smiles, test_set=test_smiles, train_set=train_smiles)

        # update metrics
        for k, v in metrics_.items():
            metrics[k] = v
        return metrics
