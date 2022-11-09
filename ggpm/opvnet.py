from ggpm.property_vae import *


class OPVNet:
    MODEL_DICT = {
        'prop': PropertyVAE,
        'prop-opt': PropOptVAE,
        'hier-prop': HierPropertyVAE,
        'hier-prop-opt': HierPropOptVAE
    }

    @staticmethod
    def get_model(name):
        return OPVNet.MODEL_DICT[name]
