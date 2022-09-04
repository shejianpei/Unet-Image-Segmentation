import torch
from imgPretreatment.imgPre import imgpre
from net.unet import unett
from utils import fitfun
from utils.optim_loss import optim_loss

model = unett()

config = {
        "batch_size": 16,
        "epochs": 40,
        "lr": 0.0001,
        "seed": 2021,
        "momentum": 0.1,
        "PATH": "./model_.pth",
        "IMG_SIZE": 256,
        "SUO_FANG_IMG_SIZE": 256,
        "ROTATION": 0.2,
    }

train_dl, test_dl = imgpre(config)
optimizer, loss_fn, exp_lr_scheduler = optim_loss(model, config)


for epoch in range(config.get("epochs")):
    fitfun.fit(epoch, model, train_dl, test_dl, optimizer, loss_fn, exp_lr_scheduler, config)

    # print("save.pth")
    torch.save(model.state_dict(), config.get("PATH"))
