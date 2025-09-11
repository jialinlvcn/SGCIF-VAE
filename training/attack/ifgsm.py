import torch

def ifgsm(data, label, model, 
        eps = 8/255, epochs=10, 
        is_tg=False, backbone=None, mu=1.0, norm='linf', **params):
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
    for _ in range(epochs):
        data = data.detach()
        data.requires_grad = True
        if backbone is not None:
            output = backbone(model(data))
        else:
            output = model(data)
        try:
            cost = loss(output, label.to(device)).to(device)
        except:
            cost = loss(output[0], label.to(device)).to(device)
        if is_tg:
            grad = torch.autograd.grad(cost, data, 
                                        retain_graph=False, 
                                        create_graph=False)[0]
        else:
            grad = - torch.autograd.grad(cost, data, 
                                        retain_graph=False, 
                                        create_graph=False)[0]
        data = data.detach() - alpha * grad.sign()
        data = torch.where(data > data + eps, 
                                data + eps, data)
        data = torch.where(data < data - eps, 
                                data - eps, data)
        data = torch.clamp(data.detach(), 0, 1)
    return data