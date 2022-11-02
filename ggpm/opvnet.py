from ggpm.property_vae import *

class OPVNet:
    MODEL_DICT = {
            'prop': PropertyVAE,
            'prop_opt': PropOptVAE,
            'hier_prop': HierPropertyVAE,
            'hier_prop_opt': HierPropOptVAE
            }

    @staticmethod
    def get_model(name):
        return OPVNet.MODEL_DICT[name]
