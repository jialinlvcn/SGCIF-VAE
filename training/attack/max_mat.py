from .cw import CW_linf
from .ifgsm import ifgsm
from .pgd import pgd
import torch

def max_mat(
    data, 
    label, 
    model, 
    eps=16/255, 
    epochs=10, 
    **params
):
    device = params.get('device', 'cpu')
    
    # Generate adversarial examples using different methods
    adv_cw = CW_linf(data, label, model, epochs=epochs, eps=eps, device=device)
    adv_ifgsm = ifgsm(data, label, model, epochs=epochs, eps=eps, device=device)
    adv_pgd = pgd(data, label, model, epochs=epochs, eps=eps, device=device)
    
    # Get model predictions for each adversarial example
    soft_cw = model(adv_cw)[0]
    soft_ifgsm = model(adv_ifgsm)[0]
    soft_pgd = model(adv_pgd)[0]
    
    # Handle case where output is a tuple
    if isinstance(soft_cw, tuple):
        soft_cw = soft_cw[0]
        soft_ifgsm = soft_ifgsm[0]
        soft_pgd = soft_pgd[0]
    
    # Get predictions for the target labels
    B, C, H, W = data.shape
    label_unsqueezed = label.unsqueeze(1).to(device)
    
    pred_cw = torch.gather(soft_cw, dim=1, index=label_unsqueezed).squeeze(1)
    pred_ifgsm = torch.gather(soft_ifgsm, dim=1, index=label_unsqueezed).squeeze(1)
    pred_pgd = torch.gather(soft_pgd, dim=1, index=label_unsqueezed).squeeze(1)
    
    # Select the most effective adversarial examples
    all_preds = torch.stack([pred_cw, pred_ifgsm, pred_pgd], dim=0)
    sel_indexs = torch.argmin(all_preds, dim=0)
    
    # Prepare indices for gathering
    sel_indexs_expanded = sel_indexs.view(1, B, 1, 1, 1).expand(1, B, C, H, W).to(device)
    
    # Combine adversarial examples based on effectiveness
    adv_stack = torch.stack([adv_cw, adv_ifgsm, adv_pgd], dim=0)
    adv_images = torch.gather(adv_stack, dim=0, index=sel_indexs_expanded).squeeze(0)
    
    return adv_images
