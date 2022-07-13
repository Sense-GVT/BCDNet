from .a_3 import a_3
from .a_3f import a_3f


def model_entry(config):

    if config['type'] not in globals():
        from prototype.spring import PrototypeHelper
        return PrototypeHelper.external_model_builder[config['type']](**config['kwargs'])

    return globals()[config['type']](**config['kwargs'])
