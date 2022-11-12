from ggpm.property_vae import *
from ggpm.property_control import *

OPTIMIZER_MAP = {
    'prop-optm': 'prop-opt',
    'hier-prop-optm': 'hier-prop-opt'
}

class OPVNet:
    MODEL_DICT = {
        'prop': PropertyVAE,
        'prop-opt': PropOptVAE,
        'hier-prop': HierPropertyVAE,
        'hier-prop-opt': HierPropOptVAE,
        'prop-optm': PropertyVAEOptimizer,
        'hier-prop-optm': HierPropertyVAEOptimizer
    }

    @staticmethod
    def get_model(name):
        return OPVNet.MODEL_DICT[name]

    @staticmethod
    def get_control_model(name):
        assert name in ['prop-optm', 'hier-prop-optm'], ValueError('{} is not a valid optimizer choice.'.format(name))
        core_class = OPTIMIZER_MAP[name]
        return OPVNet.get_model(core_class), OPVNet.MODEL_DICT[name]

