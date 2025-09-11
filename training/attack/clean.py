import torch
import math

loss = torch.nn.CrossEntropyLoss()
def clean(data, label, model, 
        eps = 8/255, epochs=10, 
        is_tg=False, backbone=None, **params):
        
    return data