from ggpm.property_vae import *
from evaluation.chemberta_pr2 import *

class OPVNet:
    MODEL_DICT = {
            'prop': PropertyVAE,
            'prop-opt': PropOptVAE,
            'hier-prop': HierPropertyVAE,
            'hier-prop-opt': HierPropOptVAE,
            'pr2-single': ChemBertaForSinglePR2,
            'pr2-two': ChemBertaForTwoPR2,
            'pr2-two-split': ChemBertaForTwoSplitPR2,
            }

    @staticmethod
    def get_model(name):
        return OPVNet.MODEL_DICT[name]
