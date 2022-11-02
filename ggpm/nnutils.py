import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import AllChem
from rdkit import DataStructs
from collections import OrderedDict

is_cuda = torch.cuda.is_available()
device = torch.device('cuda:0') if is_cuda else torch.device('cpu')


def copy_model(tbc_model, tc_model, path, w_property=False):
    # Function to copy encoder-decoder model w/ property-optimizer

    # load tc_model: encoder & decoder
    # and load tbc_model
    tc_model.load_state_dict(torch.load(path, map_location=device))
    tc_model_dict = {'encoder': tc_model.encoder.state_dict(),
                     'decoder': tc_model.decoder.state_dict()}
    tbc_model_dict = {'encoder': tbc_model.encoder.state_dict(),
                      'decoder': tbc_model.decoder.state_dict()}

    # load property_optimizer if required
    if w_property is True:
        tc_model_dict['property_optim'] = tc_model.property_optim.state_dict()
        tbc_model_dict['property_optim'] = tbc_model.property_optim.state_dict()

    # filter pretrained dict
    tc_model_dict = {seq_k: {k: v for k, v in tc_dict.items() if
                             (k in tbc_model_dict[seq_k]) and (tbc_model_dict[seq_k][k].shape == tc_dict[k].shape)}
                     for seq_k, tc_dict in tc_model_dict.items()}

    # copy to tbc_model
    tbc_model.encoder.load_state_dict(tc_model_dict['encoder'])
    tbc_model.decoder.load_state_dict(tc_model_dict['decoder'])
    if w_property is True:
        tbc_model.property_optim.load_state_dict(tc_model_dict['property_optim'])

    print("Successfully copied the model with property_head={}.".format(w_property))
    return tbc_model


def copy_encoder(tbc_model, tc_model, path):
    # Function to copy encoder only

    # load tc_model
    tc_model.load_state_dict(torch.load(path, map_location=device))
    tc_model_dict = tc_model.encoder.state_dict()

    # filter pretrained dict
    model_dict = tbc_model.encoder.state_dict()
    tc_model_dict = {k: v for k, v in tc_model_dict.items() if
                     (k in model_dict) and (model_dict[k].shape == tc_model_dict[k].shape)}

    # load model dict
    # model_dict.update(tc_model_dict)
    tbc_model.encoder.load_state_dict(tc_model_dict)

    print("Successfully copied encoder.")
    return tbc_model


def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)


def avg_pool(all_vecs, scope, dim):
    size = create_var(torch.Tensor([le for _, le in scope]))
    return all_vecs.sum(dim=dim) / size.unsqueeze(-1)


def get_accuracy_bin(scores, labels):
    preds = torch.ge(scores, 0).long()
    acc = torch.eq(preds, labels).float()
    return torch.sum(acc) / labels.nelement()


def get_accuracy(scores, labels):
    _, preds = torch.max(scores, dim=-1)
    acc = torch.eq(preds, labels).float()
    return torch.sum(acc) / labels.nelement()


def get_accuracy_sym(scores, labels):
    max_scores, max_idx = torch.max(scores, dim=-1)
    lab_scores = scores[torch.arange(len(scores)), labels]
    acc = torch.eq(lab_scores, max_scores).float()
    return torch.sum(acc) / labels.nelement()


def stack_pad_tensor(tensor_list):
    max_len = max([t.size(0) for t in tensor_list])
    for i, tensor in enumerate(tensor_list):
        pad_len = max_len - tensor.size(0)
        tensor_list[i] = F.pad(tensor, (0, 0, 0, pad_len))
    return torch.stack(tensor_list, dim=0)


def create_pad_tensor(alist):
    max_len = max([len(a) for a in alist]) + 1
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([0] * pad_len)
    return torch.IntTensor(alist)


def zip_tensors(tup_list, is_concat=False):
    res = []
    tup_list = zip(*tup_list)
    for a in tup_list:
        if type(a[0]) is int:
            res.append(torch.LongTensor(a))  # .cuda())
        else:
            res.append(torch.cat(a, dim=0) if is_concat else torch.stack(a, dim=0))
    return res


def index_scatter(sub_data, all_data, index):
    d0, d1 = all_data.size()
    buf = torch.zeros_like(all_data).scatter_(0, index.repeat(d1, 1).t(), sub_data)
    mask = torch.ones(d0, device=all_data.device).scatter_(0, index, 0)
    return all_data * mask.unsqueeze(-1) + buf


def hier_topk(cls_scores, icls_scores, vocab, topk):
    batch_size = len(cls_scores)
    cls_scores = F.log_softmax(cls_scores, dim=-1)
    cls_scores_topk, cls_topk = cls_scores.topk(topk, dim=-1)
    final_topk = []
    for i in range(topk):  # permuate topk fragments and topk possible fragment configs
        clab = cls_topk[:, i]
        mask = vocab.get_mask(clab)
        masked_icls_scores = F.log_softmax(icls_scores + mask, dim=-1)
        icls_scores_topk, icls_topk = masked_icls_scores.topk(topk, dim=-1)
        topk_scores = cls_scores_topk[:, i].unsqueeze(-1) + icls_scores_topk  # topk_scores = [bs, top_k]
        final_topk.append((topk_scores, clab.unsqueeze(-1).expand(-1, topk), icls_topk))

    topk_scores, cls_topk, icls_topk = zip(*final_topk)
    topk_scores = torch.cat(topk_scores, dim=-1)  # [bs, top_k * top_k]
    cls_topk = torch.cat(cls_topk, dim=-1)
    icls_topk = torch.cat(icls_topk, dim=-1)

    topk_scores, topk_index = topk_scores.topk(topk, dim=-1)  # get top_k scores of combination
    batch_index = cls_topk.new_tensor([[i] * topk for i in range(batch_size)])
    cls_topk = cls_topk[batch_index, topk_index]
    icls_topk = icls_topk[batch_index, topk_index]
    return topk_scores, cls_topk.tolist(), icls_topk.tolist()


def to_cuda(inputs):
    # Function to convert to cuda tensors if cuda cores exist
    return inputs.to(device)


def property_grad_optimize(vecs, outputs, targets, latent_lr):
    # Function to perform gradient descent/ascent depending on
    # the value comparison between output and target
    # if output >= target, descent else ascent

    # create mask of -1 and 1
    # in shape of vecs
    mask = -2 * (outputs >= targets).float()
    mask = torch.ones_like(mask) + mask  # add mask of selected -2s + mask of all 1s
    mask = torch.unsqueeze(mask, dim=-1)  # expand dim for broadcasting

    return vecs + latent_lr * vecs.grad * mask


def get_tanimoto_dist(mol_x, mol_y, radius=3, n_bits=2048):
    # Measure the Tanimoto coefficient based on Morgan fingerprints
    fp_x = AllChem.GetMorganFingerprintAsBitVect(mol_x, radius, nBits=n_bits)
    fp_y = AllChem.GetMorganFingerprintAsBitVect(mol_y, radius, nBits=n_bits)
    return round(DataStructs.TanimotoSimilarity(fp_x, fp_y), 3)


def get_frechet_dist(mol_x, mol_y, radius=3, n_bits=2048):
    # Measure the Frechet similarity based on Morgan fingerprints
    fp_x = AllChem.GetMorganFingerprintAsBitVect(mol_x, radius, nBits=n_bits)
    fp_y = AllChem.GetMorganFingerprintAsBitVect(mol_y, radius, nBits=n_bits)
    # TO-DO: add frechet_dist formula
    return None


def rename_optimizer_state_keys(model_state_dict):
    new_state_dict = OrderedDict()

    for k, v in model_state_dict.items():
        if 'property_optim' in k:
            name = k[:26] + '.linear' + k[26:]
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def make_tensor(x):
    if not isinstance(x, torch.Tensor):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        x = torch.tensor(x)

    return to_cuda(x)


def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    tree_tensors = [make_tensor(x).long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors
