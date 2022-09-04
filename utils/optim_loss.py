import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import wandb


def optim_loss(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr"))
    loss_fn = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=config.get("momentum"))
    wandb.log({"lr": config.get("lr")})
    return optimizer, loss_fn, exp_lr_scheduler
