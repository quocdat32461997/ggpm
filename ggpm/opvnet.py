from ggpm.property_vae import *
import evaluation.schnet as schnet

class OPVNet:
    MODEL_DICT = {
            'prop': PropertyVAE,
            'prop-opt': PropOptVAE,
            'hier-prop': HierPropertyVAE,
            'hier-prop-opt': HierPropOptVAE,
            'schnet': schnet.SchNetwork
            }

    @staticmethod
    def get_model(name):
        return OPVNet.MODEL_DICT[name]
