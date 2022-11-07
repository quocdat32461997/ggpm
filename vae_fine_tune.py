import torch
import torch.nn as nn
import torch.optim as torch_optim
import torch.optim.lr_scheduler as lr_scheduler

import math, sys
import argparse

from ggpm import *
from configs import *
from torchtools import EarlyStopping
import rdkit

from ggpm.property_vae import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

# get path to config
parser = argparse.ArgumentParser()
parser.add_argument('--path-to-config', required=True)
parser.add_argument('--model-finetune-type', required=True)
parser.add_argument('--model-pretrained-type', required=True)
args = parser.parse_args()

# get configs
configs = Configs(path=args.path_to_config)

vocab = [x.strip("\r\n ").split() for x in open(configs.vocab_)]
MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
configs.vocab = PairVocab([(x, y) for x, y, _ in vocab], cuda=False)

# save configs
configs.to_json(configs.save_dir + '/configs.json')

# load model
#HierPropOptVAE
model_finetune_class = OPVNet.get_model(args.model_finetune_type)
model_pretrained_class = OPVNet.get_model(args.model_pretrained_type)
model = to_cuda(model_finetune_class(configs))

# load saved encoder only
if configs.saved_model:
    # load tc_model
    if configs.load_encoder_only is True:
        model = copy_encoder(model, model_pretrained_class(configs), configs.saved_model)
        print('Successfully copied encoder weights.')
    else: # default to load entire encoder-decoder model
        model = copy_model(model, model_pretrained_class(configs), configs.saved_model, \
            w_property=configs.load_property_head)
        print('Successfully copied encoder-decoder weights.')

else:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

if configs.load_epoch >= 0:
    model.load_state_dict(torch.load(configs.save_dir + "/model." + str(configs.load_epoch)))

print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = torch_optim.Adam(model.parameters(), lr=configs.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, configs.anneal_rate)
if configs.early_stopping:
    early_stopping = EarlyStopping(patience=5, verbose=True, delta=0.01, path=configs.save_dir + '/model.{}'.format('best'))

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = 0
beta = configs.beta
metrics = {}
num_loss_clip = 5
loss_clip_break = False

for epoch in range(configs.load_epoch + 1, configs.epoch):
    dataset = DataFolder(configs.data, configs.batch_size)

    for batch in dataset:
        total_step += 1
        model.zero_grad()
        model.train()
        loss, metrics_, loss_clipped = model(*batch, beta=beta)

        # backprop
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), configs.clip_norm)
        optimizer.step()
        if loss_clipped:
            num_loss_clip -= 1
        if num_loss_clip <= 0:
            loss_clip_break = True
            break

        # accumulate metrics
        for k, v in metrics_.items():
            metrics[k] = v if not k in metrics else metrics[k] + v

        if total_step % configs.print_iter == 0:
            metrics = {k: v / configs.print_iter for k,v in metrics.items()}
            print("[%d] Beta: %.3f, PNorm: %.2f, GNorm: %.2f" % (
                total_step, beta,  param_norm(model), grad_norm(model))) # print step

            # print metrics
            print(', '.join([k + ': %.3f' % v for k, v in metrics.items()]))
            sys.stdout.flush()

            # reset metrics
            metrics = {}

        if configs.save_iter >= 0 and total_step % configs.save_iter == 0:
            n_iter = total_step // configs.save_iter - 1
            torch.save(model.state_dict(), configs.save_dir + "/model." + str(n_iter))
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])

        # evaluate
        if total_step % configs.eval_iter == 0:
            model.eval()
            val_loss, val_metrics = [], {}
            with torch.no_grad():
                val_dataset = DataFolder(configs.val_data, configs.batch_size)
                for batch in val_dataset:
                    loss, metrics_, _ = model(*batch, beta=beta)

                    val_loss.append(loss.item())
                    for k, v in metrics_.items():
                        val_metrics[k] = v if not k in val_metrics else val_metrics[k] + v

            # average val loss & metrics
            n = len(val_loss)
            val_loss = sum(val_loss) / n
            val_metrics = {k: v / n for k, v in val_metrics.items()}
            # print metrics
            print("[%d] Beta: %.3f, PNorm: %.2f, GNorm: %.2f" % (
                total_step, beta, param_norm(model), grad_norm(model)))  # print step
            print("[%d] " % total_step, ', '.join([k + ': %.3f' % v for k, v in val_metrics.items()]))
            sys.stdout.flush()

            del val_dataset
            # update early_stopping
            if configs.early_stopping:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    break

    if loss_clip_break:
        print('Stop due to loss_clip_break')
        break
    if configs.early_stopping and early_stopping.early_stop:
        print('Stop: early stopping')
        break

    del dataset
    if configs.save_iter == -1:
        torch.save(model.state_dict(), configs.save_dir + "/model." + str(epoch))
        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_lr()[0])
