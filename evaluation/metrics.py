import torch
from rdkit.Chem import MolFromSmiles, AllChem
from rdkit.Chem.Descriptors import MolWt
import moses
from torchmdnet.models.model import load_model

from ggpm import *


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
