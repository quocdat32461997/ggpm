import os
import os.path as osp
import torch
import moses
import rdkit
from torch.nn import functional as F
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data import download_url

from ggpm import *
from ggpm import OPVNet
#from torch_geometric.nn import DimeNet, DimeNetPlusPlus
#from torch_geometric.datasets import QM9

DIMNET_URL = 'https://github.com/klicperajo/dimenet/raw/master/pretrained/dimenet'
DIMENET_PP_URL = 'https://raw.githubusercontent.com/gasteigerjo/dimenet/master/pretrained/dimenet_pp'

qm9_target_dict = {
    0: 'mu',
    1: 'alpha',
    2: 'homo',
    3: 'lumo',
    5: 'r2',
    6: 'zpve',
    7: 'U0',
    8: 'U',
    9: 'H',
    10: 'G',
    11: 'Cv',
}

"""
class SimpleDimeNet(DimeNet):
    @classmethod
    def from_qm9_pretrained(cls, root, target: int):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        import tensorflow as tf

        assert 0 <= target <= 12 and target != 4
        root = osp.expanduser(osp.normpath(root))
        path = osp.join(root, 'pretrained_dimenet', qm9_target_dict[target])

        makedirs(path)
        url = f'{cls.url}/{qm9_target_dict[target]}'

        if not osp.exists(osp.join(path, 'checkpoint')):
            download_url(f'{url}/checkpoint', path)
            download_url(f'{url}/ckpt.data-00000-of-00002', path)
            download_url(f'{url}/ckpt.data-00001-of-00002', path)
            download_url(f'{url}/ckpt.index', path)

        path = osp.join(path, 'ckpt')
        reader = tf.train.load_checkpoint(path)

        model = cls(
            hidden_channels=128,
            out_channels=1,
            num_blocks=6,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
        )

        def copy_(src, name, transpose=False):
            init = reader.get_tensor(f'{name}/.ATTRIBUTES/VARIABLE_VALUE')
            init = torch.from_numpy(init)
            if name[-6:] == 'kernel':
                init = init.t()
            src.data.copy_(init)

        copy_(model.rbf.freq, 'rbf_layer/frequencies')
        copy_(model.emb.emb.weight, 'emb_block/embeddings')
        copy_(model.emb.lin_rbf.weight, 'emb_block/dense_rbf/kernel')
        copy_(model.emb.lin_rbf.bias, 'emb_block/dense_rbf/bias')
        copy_(model.emb.lin.weight, 'emb_block/dense/kernel')
        copy_(model.emb.lin.bias, 'emb_block/dense/bias')

        for i, block in enumerate(model.output_blocks):
            copy_(block.lin_rbf.weight, f'output_blocks/{i}/dense_rbf/kernel')
            for j, lin in enumerate(block.lins):
                copy_(lin.weight, f'output_blocks/{i}/dense_layers/{j}/kernel')
                copy_(lin.bias, f'output_blocks/{i}/dense_layers/{j}/bias')
            copy_(block.lin.weight, f'output_blocks/{i}/dense_final/kernel')

        for i, block in enumerate(model.interaction_blocks):
            copy_(block.lin_rbf.weight, f'int_blocks/{i}/dense_rbf/kernel')
            copy_(block.lin_sbf.weight, f'int_blocks/{i}/dense_sbf/kernel')
            copy_(block.lin_kj.weight, f'int_blocks/{i}/dense_kj/kernel')
            copy_(block.lin_kj.bias, f'int_blocks/{i}/dense_kj/bias')
            copy_(block.lin_ji.weight, f'int_blocks/{i}/dense_ji/kernel')
            copy_(block.lin_ji.bias, f'int_blocks/{i}/dense_ji/bias')
            copy_(block.W, f'int_blocks/{i}/bilinear')
            for j, layer in enumerate(block.layers_before_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/bias')
            copy_(block.lin.weight, f'int_blocks/{i}/final_before_skip/kernel')
            copy_(block.lin.bias, f'int_blocks/{i}/final_before_skip/bias')
            for j, layer in enumerate(block.layers_after_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/bias')

        return model

class SimpleDimeNetPP(DimeNetPlusPlus):
    @classmethod
    def from_qm9_pretrained(cls, root: str, target: int):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        import tensorflow as tf

        assert target >= 0 and target <= 12 and not target == 4

        root = osp.expanduser(osp.normpath(root))
        path = osp.join(root, 'pretrained_dimenet_pp', qm9_target_dict[target])

        makedirs(path)
        url = f'{cls.url}/{qm9_target_dict[target]}'

        if not osp.exists(osp.join(path, 'checkpoint')):
            download_url(f'{url}/checkpoint', path)
            download_url(f'{url}/ckpt.data-00000-of-00002', path)
            download_url(f'{url}/ckpt.data-00001-of-00002', path)
            download_url(f'{url}/ckpt.index', path)

        path = osp.join(path, 'ckpt')
        reader = tf.train.load_checkpoint(path)

        # Configuration from DimeNet++:
        # https://github.com/gasteigerjo/dimenet/blob/master/config_pp.yaml
        model = cls(
            hidden_channels=128,
            out_channels=1,
            num_blocks=4,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
            max_num_neighbors=32,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
        )

        def copy_(src, name, transpose=False):
            init = reader.get_tensor(f'{name}/.ATTRIBUTES/VARIABLE_VALUE')
            init = torch.from_numpy(init)
            if name[-6:] == 'kernel':
                init = init.t()
            src.data.copy_(init)

        copy_(model.rbf.freq, 'rbf_layer/frequencies')
        copy_(model.emb.emb.weight, 'emb_block/embeddings')
        copy_(model.emb.lin_rbf.weight, 'emb_block/dense_rbf/kernel')
        copy_(model.emb.lin_rbf.bias, 'emb_block/dense_rbf/bias')
        copy_(model.emb.lin.weight, 'emb_block/dense/kernel')
        copy_(model.emb.lin.bias, 'emb_block/dense/bias')

        for i, block in enumerate(model.output_blocks):
            copy_(block.lin_rbf.weight, f'output_blocks/{i}/dense_rbf/kernel')
            copy_(block.lin_up.weight,
                  f'output_blocks/{i}/up_projection/kernel')
            for j, lin in enumerate(block.lins):
                copy_(lin.weight, f'output_blocks/{i}/dense_layers/{j}/kernel')
                copy_(lin.bias, f'output_blocks/{i}/dense_layers/{j}/bias')
            copy_(block.lin.weight, f'output_blocks/{i}/dense_final/kernel')

        for i, block in enumerate(model.interaction_blocks):
            copy_(block.lin_rbf1.weight, f'int_blocks/{i}/dense_rbf1/kernel')
            copy_(block.lin_rbf2.weight, f'int_blocks/{i}/dense_rbf2/kernel')
            copy_(block.lin_sbf1.weight, f'int_blocks/{i}/dense_sbf1/kernel')
            copy_(block.lin_sbf2.weight, f'int_blocks/{i}/dense_sbf2/kernel')

            copy_(block.lin_ji.weight, f'int_blocks/{i}/dense_ji/kernel')
            copy_(block.lin_ji.bias, f'int_blocks/{i}/dense_ji/bias')
            copy_(block.lin_kj.weight, f'int_blocks/{i}/dense_kj/kernel')
            copy_(block.lin_kj.bias, f'int_blocks/{i}/dense_kj/bias')

            copy_(block.lin_down.weight,
                  f'int_blocks/{i}/down_projection/kernel')
            copy_(block.lin_up.weight, f'int_blocks/{i}/up_projection/kernel')

            for j, layer in enumerate(block.layers_before_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/bias')

            copy_(block.lin.weight, f'int_blocks/{i}/final_before_skip/kernel')
            copy_(block.lin.bias, f'int_blocks/{i}/final_before_skip/bias')

            for j, layer in enumerate(block.layers_after_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/bias')

        return model
 """       
class Metrics:
    def __init__(self, batch_size=512, ks=[50, 500], num_worker=2, device='cpu'):
        self.ks = ks
        self.num_worker = num_worker
        self.device = device
        self.batch_size = batch_size

        # regressor: DimeNetPlusPlus by default
        #dataset = QM9('ckpt')
        #self.homo_model = to_cuda(SimpleDimeNetPP.from_qm9_pretrained(path='ckpt', target=2))
        #self.homo_model.eval()
        #self.lumo_model = to_cuda(SimpleDimeNetPP.from_qm9_pretraind(path='ckpt', target=3))
        #self.lumo_model.eval()

    def get_recon_n_sample_metrics(self, output_set, test_set, train_set=None):
        return moses.get_all_metrics(gen=output_set, k=self.ks, n_jobs=self.num_worker,
                                     device=self.device, batch_size=self.batch_size,
                                     test=test_set, train=train_set)

    def _evaluate_gap(self, model_homo, model_lumo, data, batch_size=32):
        model_homo.eval()
        model_lumo.eval()
        with torch.no_grad():
            n_batch = len(data) // batch_size
            n_batch = (n_batch + 1) if (len(data) % batch_size) != 0 else n_batch
            dg_MSE = 0.
            lg_MSE = 0.
            cb_MSE = 0.

            for i in range(n_batch):
                batch_hg, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat, y = data.next_batch(
                    batch_size, 'gap')

                dg_node_feat_discrete = dg_node_feat_discrete.to(device)
                lg_node_feat_continuous = lg_node_feat_continuous.to(device)
                lg_node_feat_discrete = lg_node_feat_discrete.to(device)
                dg_edge_feat = dg_edge_feat.to(device)
                lg_edge_feat = lg_edge_feat.to(device)

                y = y.to(device)

                dg_y_homo, lg_y_homo, y_homo, _ = self.homo_model(
                    batch_hg, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat)
                dg_y_lumo, lg_y_lumo, y_lumo, _ = self.lumo_model(
                    batch_hg, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat)

                # get the prediction score
                dg_MSE += F.mse_loss((dg_y_lumo - dg_y_homo).squeeze(), y, reduction='sum').item()
                lg_MSE += F.mse_loss((lg_y_lumo - lg_y_homo).squeeze(), y, reduction='sum').item()
                cb_MSE += F.mse_loss(y_lumo - y_homo, y, reduction='sum').item()

            return dg_MSE / len(data), lg_MSE / len(data), cb_MSE / len(data)

    def get_optimization_metrics(self, output_smiles):
        pass
