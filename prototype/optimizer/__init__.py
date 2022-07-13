import linklink as link

from torch.optim import SGD, RMSprop, Adadelta, Adagrad, Adam, AdamW  # noqa F401
from .lars import LARS  # noqa F401
from .fp16_optim import FP16SGD, FP16RMSprop  # noqa F401
try:
    from linklink.optim import FusedFP16SGD
except ModuleNotFoundError:
    print('import FusedFP16SGD failed, linklink version should >= 0.1.6')
    FusedFP16SGD = None


def optim_entry(config):
    rank = link.get_rank()
    if config['type'] == 'FusedFP16SGD' and FusedFP16SGD is None:
        raise RuntimeError('FusedFP16SGD is disabled due to linklink version, try using other optimizers')
    if config['type'] == 'FusedFP16SGD' and rank > 0:
        config['kwargs']['verbose'] = False
    return globals()[config['type']](**config['kwargs'])
