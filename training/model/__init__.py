from model.standard import opt_backbone
from model.sgcif.sg import SG_VGNN
from .bagnet.model import BottleneckNoBn, SmallBagNet, TinyBagNet, BagNet
from .psgan.model import PSGANModel
from .ape.model import ape
from .cd.model import CDVAE


def get_model(model_name, **kwargs):
    if model_name.lower() == "sgcif":
        return SG_VGNN(**kwargs)
    elif model_name.lower() == "bagnet":
        return SmallBagNet(BottleneckNoBn, **kwargs)
    elif model_name.lower() == "psgan":
        return PSGANModel(**kwargs)
    elif model_name.lower() == "ape":
        return ape(**kwargs)
    elif model_name.lower() == "cdvae":
        return CDVAE(**kwargs)
    elif model_name.lower() == "standard":
        return opt_backbone(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not recognized.")


__all__ = [
    "get_model",
    "BottleneckNoBn",
    "SmallBagNet",
    "TinyBagNet",
    "BagNet",
    "PSGANModel",
    "ape",
    "CDVAE",
    "opt_backbone",
]
