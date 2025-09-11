from .cw import CW_linf
from .ifgsm import ifgsm
from .pgd import pgd
import torch
import random

def rand_mat(
    data, 
    label, 
    model, 
    eps=16/255, 
    epochs=10, 
    **params
):
    device = params.get('device', 'cpu')
    
    # Generate adversarial examples using different methods
    attack_methods = [
        lambda: CW_linf(data, label, model, epochs=epochs, eps=eps, device=device),
        lambda: ifgsm(data, label, model, epochs=epochs, eps=eps, device=device),
        lambda: pgd(data, label, model, epochs=epochs, eps=eps, device=device)
    ]
    selected_attack = random.choice(attack_methods)
    adv_images = selected_attack()
    
    return adv_images
