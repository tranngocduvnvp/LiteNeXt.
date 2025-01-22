import torch
import torch.nn as nn
from Metrics.losses import SoftDiceLoss, FocalLoss, TverskyLoss
from Metrics.metrics import DiceScore
from Datasets.dataloader import get_dataloader_lung
import importlib


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    train_dataloader, val_dataloader, test_dataloader = get_dataloader_lung(args.root, batch_size=args.batch_size)

    Dice_loss = SoftDiceLoss()
    BCE_loss = nn.BCELoss()
    Tverskyloss = TverskyLoss()
    Focalloss = FocalLoss()
    Ssim = nn.MSELoss()
    Smooth = nn.SmoothL1Loss()
    loss_fun = {'Dice_loss':Dice_loss, "BCE_loss":BCE_loss, "TverskyLoss":Tverskyloss, "FocalLoss":Focalloss,\
                "Ssim":Ssim, "Smooth":Smooth}

    perf = DiceScore()

    #===================== Model ===================================================
    # Chọn mô hình
    model_mapping = {
    "Attentionunet": "Models.Attentionunet",
    "Doubleunet": "Models.Doubleunet",
    "Fcn": "Models.Fcn",
    "Unext": "Models.Unext",
    "Unet": "Models.Unet",
    "LiteNeXt": "Models.LiteNeXt",
}

    model_module = importlib.import_module(model_mapping[args.model_name["name"]])
    model = getattr(model_module, args.model_name["version"])()

    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)

    #===================== Optimizer ===================================================
    if args.optim == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    elif args.optim == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "SparseAdam":
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.lr)
    elif args.optim == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
    elif args.optim == "ASGD":
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
    elif args.optim == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=args.lr)
    elif args.optim == "NAdam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr)
    elif args.optim == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
    elif args.optim == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == "Rprop":
        optimizer = torch.optim.Rprop(model.parameters(), lr=args.lr)
    #===================================================================================

    if args.lrs == "true":
        if args.type_lr == "LROnP":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optimizer, mode="max", patience=30, factor=0.75, min_lr=args.lrs_min, verbose=True)
        elif args.type_lr == "StepLR":
            print("Using StepLR")
            scheduler = torch.optim.lr_scheduler.StepLR(
                  optimizer, step_size=100, gamma=0.25, verbose=False)
        elif args.type_lr == "MultiStepLR":
            print("Using MultiStepLR")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                  optimizer, milestones=[10, 20, 30, 60], gamma=0.5, verbose=False)


    if args.checkpoint_path == None:
        checkpoint = {"test_measure_mean":None, "epoch":0}
    else:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (device, train_dataloader, val_dataloader, test_dataloader, 
            perf, model, optimizer, checkpoint, scheduler, loss_fun)



if __name__ == "__main__":
    ( device, train_dataloader, val_dataloader, test_dataloader,
     perf, model, optimizer, checkpoint, scheduler, loss_fun) = build(args)