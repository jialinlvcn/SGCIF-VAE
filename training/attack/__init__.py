from .pgd import pgd
from .ifgsm import ifgsm
from .cw import CW_linf as cw
from .clean import clean
from .max_mat import max_mat
from .rand_mat import rand_mat
from .multi_mat import multi_mat

attack_box = {
    "pgd": pgd,
    "cw": cw,
    "ifgsm": ifgsm,
    "clean": clean,
    "max_mat": max_mat,
    "rand_mat": rand_mat,
    "multi_mat": multi_mat,
}

eval_box = {
    "clean": (clean, 0, 0),
    "pgd16": (pgd, 16 / 255, 10),
    "pgd32": (pgd, 32 / 255, 30),
    "ifgsm16": (ifgsm, 16 / 255, 10),
    "ifgsm32": (ifgsm, 32 / 255, 30),
    "cw16": (cw, 16 / 255, 10),
    "cw32": (cw, 32 / 255, 30),
}
