from .cw import CW_linf
from .ifgsm import ifgsm
from .pgd import pgd
import torch

def multi_mat(
    data, 
    label, 
    model, 
    eps=16/255, 
    epochs=10, 
    **params
):
    device = params.get('device', 'cpu')
    
    # Generate adversarial examples using different methods
    adv_cw_delta = CW_linf(data, label, model, epochs=epochs, eps=eps, device=device) - data.to(device)
    adv_ifgsm_delta = ifgsm(data, label, model, epochs=epochs, eps=eps, device=device) - data.to(device)
    adv_pgd_delta = pgd(data, label, model, epochs=epochs, eps=eps, device=device) - data.to(device)
    
    delta = adv_cw_delta + adv_ifgsm_delta + adv_pgd_delta
    delta = torch.clamp(delta, min=-eps, max=eps)
    delta = (torch.clamp(data.to(device) + delta, min=0, max=1) - data.to(device)).detach()

    adv_images = data.to(device) + delta
    
    return adv_images
