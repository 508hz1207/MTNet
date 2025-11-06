from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_config
from dataset import RoadDataset, graph_collate_fn
from model import MTRoad
import os
import matplotlib.pyplot as plt
import matplotlib
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table

matplotlib.use('Agg')
os.environ["CUDA_VISIBLE_DEVICES"]='0'

if __name__ == "__main__":
    parser=ArgumentParser()

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    # parser.add_argument('--config-path',type=str,default='config/config_512_massachusetts.yaml')
    parser.add_argument('--config-path',type=str,default='config/config_512_CHN6_CUG.yaml')


    args = parser.parse_args()

    config = load_config(args.config_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


    net = MTRoad(config).to(device)


    train_ds, val_ds= RoadDataset(config, is_train=True,type_is="train"), RoadDataset(config, is_train=False,type_is="val")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )
    optimizer,scheduler=net.configure_optimizers()
    bce_loss_plt=[]
    dice_loss_plt=[]
    train_loss_plt=[]
    lear_rate=[]

    precision=[]
    recall=[]
    iou_p=[]
    f1=[]

    val_best_metrics = -0.5
    val_best_loss = float("+inf")
    no_optim = 0

    if config.LOAD_OPT_SCH:
        path = config.LOAD_TOTAL_WEIGHT_PATH
        checkpoint = torch.load(path, map_location="cpu")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    if config.LOAD_OPT_SCH:
        start_epoch=checkpoint.get("epoch",0)+1
    else:
        start_epoch=0


    for epoch in range(start_epoch,config.TRAIN_EPOCHS):
        train_loss=net.train_one_epoch(bce_loss_plt,dice_loss_plt,train_loss_plt,lear_rate,optimizer=optimizer,data_loader=train_loader,device=device,epoch=epoch)
        scheduler.step()
        old_lr = optimizer.param_groups[0]["lr"]

        validation_loss=net.validation_one_epoch(precision,recall,f1,iou_p,optimizer=optimizer,scheduler=scheduler,data_loader=val_loader,device=device,epoch=epoch)

    bce_loss_plt=np.array(bce_loss_plt)
    dice_loss_plt=np.array(dice_loss_plt)
    train_loss_plt=np.array(train_loss_plt)
    x=list(range(bce_loss_plt.shape[0]))

    plt.figure(0)
    plt.plot(x, train_loss_plt, color="red", label="Total Loss")  # （BCE+Dice)
    plt.plot(x, bce_loss_plt, color="green", label="Mask Loss")  # （BCE)
    plt.plot(x, dice_loss_plt, color="yellow", label="Surface Loss")  # （Dice)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(config.ROAD_RESULT_PATH,"loss.png"))

