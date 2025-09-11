import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class AverageMeter(object):
    def __init__(self, string=""):
        self.name = string
        self.reset()

    def reset(self):
        self.avg = 0
        self.rsum = 0
        self.count = 0
        self.matrix = [[0] * 10 for _ in range(10)]

    def update(self, pred, gt):
        n = len(gt)
        val = (pred == gt).sum().item()
        self.count += n
        self.rsum += val
        self.avg = self.rsum / self.count * 100
        for i, j in zip(gt, pred):
            self.matrix[i][j] += 1

    def __str__(self):
        return f"the accuracy of {self.name} is {self.avg * 100} %"

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight)

class NaNError(Exception):
    pass

def check_model_nan(model):
    for param in model.parameters():
        if torch.isnan(param).any():
            raise NaNError("have nan in model parameters")
    return False


def check_model_grad_nan(model):
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            raise NaNError("have nan in model grad")
    return False

def trades(model, x, x_adv, y, mu=1.0):
    out = model(x_adv)
    if isinstance(out, tuple):
        logits_natural = model(x)[0]
        logits_robust = model(x_adv)[0]
    else:
        logits_natural = model(x)
        logits_robust = model(x_adv)
    loss_natural = F.cross_entropy(logits_natural, y)

    bs = x.size(0)
    criterion_kl = nn.KLDivLoss(reduction="sum")
    loss_robust = (1.0 / bs) * criterion_kl(
        F.log_softmax(logits_robust, dim=1), F.softmax(logits_natural, dim=1)
    )
    loss = loss_natural + loss_robust / mu
    return loss, out


def advt(model, x_adv, y):
    out = model(x_adv)
    if isinstance(out, tuple):
        logits_robust = out[0]
    else:
        logits_robust = out
    cross_entropy = F.cross_entropy(logits_robust, y)
    return cross_entropy, out

def train_one_epoch(model, loader, loss_function, attack_fun, optimizer, device, logger, eps=16 / 255, epochs=10):
    model = model.to(device)
    import time
    start_time = time.time()
    total_loss = 0.0
    num_batches = 0
    for _, (x, y, _) in tqdm(enumerate(loader), total=len(loader), desc="Training"):
        # generate adversarial examples
        x_adv = x.clone().detach()
        x_adv = attack_fun(x_adv, y, model, epochs=10, eps=16 / 255, device=device)
        x, y, x_adv = x.to(device), y.to(device), x_adv.to(device)

        # start training
        model.train()
        optimizer.zero_grad()

        loss, _ = loss_function(model, x, x_adv, y)
        total_loss += loss.item()
        num_batches += 1
        loss.backward()
        optimizer.step()

    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Epoch finished. Total loss: {total_loss:.4f}, Average loss: {avg_loss:.4f}, Batches: {num_batches}, Time: {elapsed_time:.2f} sec")

def evaluate(model, loader, attack_fun, device):
    model = model.to(device)
    model.eval()
    avg_meter = AverageMeter("Test")
    
    for _, (x, y, _) in tqdm(enumerate(loader), total=len(loader), desc="Evaluating"):
        # generate adversarial examples
        x_adv = x.clone().detach()
        x_adv = attack_fun(x_adv, y, model, epochs=10, eps=16 / 255, device=device)
        x, y, x_adv = x.to(device), y.to(device), x_adv.to(device)

        with torch.no_grad():
            out = model(x_adv)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            preds = torch.argmax(logits, dim=1)
            avg_meter.update(preds.cpu(), y.cpu())
    return avg_meter.avg