from ggpm.property_vae import *
import evaluation.property_nets.chemberta_pr2 as chemberta_pr2
import evaluation.property_nets.schnet as schnet

class OPVNet:
    MODEL_DICT = {
            'prop': PropertyVAE,
            'prop-opt': PropOptVAE,
            'hier-prop': HierPropertyVAE,
            'hier-prop-opt': HierPropOptVAE,
            'pr2-single': chemberta_pr2.ChemBertaForSinglePR2,
            'pr2-two': chemberta_pr2.ChemBertaForTwoPR2,
            'pr2-two-split': chemberta_pr2.ChemBertaForTwoSplitPR2,
            'schnet': schnet.SchNetwork
            }

    @staticmethod
    def get_model(name):
        return OPVNet.MODEL_DICT[name]
