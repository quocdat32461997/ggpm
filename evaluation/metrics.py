import torch
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import MolWt
import moses
from ggpm.ggpm import *


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
