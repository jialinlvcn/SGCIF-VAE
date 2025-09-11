from attack import eval_box
import utils
import torch
import os
import pandas as pd

def adv_eval(model, test_loader, attack_tuple, device="cpu", is_backbone=False):
    attack_func, eps, epochs = attack_tuple
    meter = utils.AverageMeter()
    model = model.to(device)
    model.eval()
    for x, y, path_list in test_loader:
        adv_x = x.clone().detach()
        if not is_backbone:
            adv_x = attack_func(adv_x, y, model, epochs=epochs, eps=eps, device=device)
        else:
            adv_x = attack_func(
                adv_x, y, model.backbone, epochs=epochs, eps=eps, device=device
            )

        with torch.no_grad():
            x, y, adv_x = (
                x.to(device),
                y.to(device).view(-1),
                adv_x.to(device),
            )
            out = model(adv_x)
            if isinstance(out, tuple):
                out = out[0]
            else:
                out = out
            pred = torch.argmax(out, dim=1)
            meter.update(pred, y)
    return meter

def norm_eval(
    model,
    test_loader,
    save_path: str,
    device="cpu",
    is_backbone=False,
):
    result_value = []
    result_key = []
    for key, value in eval_box.items():
        meter = adv_eval(model, test_loader, value, device, is_backbone)
        result_value.append(round(meter.avg, 4))
        result_key.append(key)
    df_percentages = pd.DataFrame([result_value], columns=result_key)

    if not os.path.exists(save_path + ".csv"):
        df_percentages.to_csv(save_path + ".csv", index=False)
    else:
        df_percentages.to_csv(save_path + ".csv", mode="a", header=False, index=False)
    return result_key, result_value
