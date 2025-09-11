import torch

def pgd(data, label, model, 
        eps = 8/255, epochs=10, 
        is_tg=False, backbone=None, norm='linf', **params):
    loss = torch.nn.CrossEntropyLoss()
    if 'device' in params.keys():
        device = params['device']
    else:
        device = 'cpu'

    if model.training:
        model.eval()
    try:
        backbone.eval()
        backbone = backbone.to(device)
    except:
        pass
    alpha = eps / epochs
    model = model.to(device)
    data = data.detach().to(device)
    delta = torch.empty_like(data).uniform_(-eps, eps)
    delta = torch.clamp(data + delta, min=0, max=1).detach() - data
    for _ in range(epochs):
        delta = delta.detach()
        delta.requires_grad = True
        if backbone is not None:
            output = backbone(model(data + delta))
        else:
            output = model(data + delta)
        try:
            cost = loss(output, label.to(device)).to(device)
        except:
            cost = loss(output[0], label.to(device)).to(device)
        if is_tg:
            grad = torch.autograd.grad(cost, delta, 
                                        retain_graph=False, 
                                        create_graph=False)[0]
        else:
            grad = - torch.autograd.grad(cost, delta, 
                                        retain_graph=False, 
                                        create_graph=False)[0]
        
        if norm == 'linf':
            delta = delta.detach() - alpha * grad.sign()
            delta = torch.clamp(delta, min=-eps, max=eps)
            delta = (torch.clamp(data + delta, min=0, max=1) - data).detach()
        elif norm == 'l2':
            delta = delta.detach() - grad / (torch.sqrt(torch.sum(grad * grad, dim=(1,2,3), keepdim=True)) + 10e-8)
            delta = delta.detach() - alpha * grad.sign()
            mask = eps >= delta.view(delta.shape[0], -1).norm(2, dim=1)
            scale = delta.view(delta.shape[0], -1).norm(2, dim=1)
            scale[mask] = eps
            delta = delta * eps / scale.view(-1, 1, 1, 1)
            delta = (torch.clamp(data + delta, min=0, max=1) - data).detach()

    return data + delta