import os
import argparse
import datetime
import utils
import torch
from torch.utils.data import DataLoader
from dataset.sar_dataset import SARFolder
from model import get_model
from attack import attack_box
from torch import nn
from tqdm import tqdm

parser = argparse.ArgumentParser(description="SGCIF-VAE Training")
parser.add_argument(
    "--data_root",
    default="./sar_datasets",
    type=str,
    help="Root directory of SAR dataset",
)
parser.add_argument(
    "--task",
    default="MSTAR_SOC",
    type=str,
    help="Task name, e.g. MSTAR_SOC, MSTAR_EOC1, SAR_ACD",
)
parser.add_argument(
    "--model_name", default="psgan", type=str, help="Model name, e.g. sgcif"
)
parser.add_argument(
    "--seed", default=666, type=int, help="Random seed for reproducibility"
)
parser.add_argument(
    "--log_dir",
    default="./logs/training",
    type=str,
    help="Directory to save training logs",
)
parser.add_argument(
    "--ckpt_dir",
    default="./checkpoints",
    type=str,
    help="Directory to save model checkpoints",
)
parser.add_argument(
    "--device",
    default="cuda:1",
    type=str,
    help="Device to use for training, e.g. cuda:0 or cpu",
)
parser.add_argument(
    "--epochs", default=100, type=int, help="Total number of training epochs"
)
parser.add_argument(
    "--batch_size", default=64, type=int, help="Batch size for training"
)
parser.add_argument("--ngf", default=64, type=float)
parser.add_argument("--ndf", default=64, type=float)
parser.add_argument("--l1_lambda", default=50, type=float)
parser.add_argument("--Dfeat_lambda", default=100, type=float)
parser.add_argument(
    "--step", default=50, type=int, help="Steps interval for logging or saving model"
)
parser.add_argument("--lr", default=5e-4, type=float, help="Learning rate")
parser.add_argument(
    "--gamma", default=0.1, type=float, help="Gamma coefficient in loss function"
)
parser.add_argument(
    "--base_model",
    default="resnet",
    type=str,
    help="Backbone network type, e.g. resnet, vig",
)
parser.add_argument(
    "--attack",
    default="pgd",
    type=str,
    help="Adversarial training method, e.g. pgd, fgsm",
)
parser.add_argument(
    "--at_mode",
    default="at",
    type=str,
    help="Training mode, e.g. at (adversarial training), normal",
)
args = parser.parse_args()

loss_bce = nn.BCELoss()
loss_mse = nn.MSELoss()

utils.seed_everything(args.seed)


def test(model, testloader, attack_fun, device, eps=16 / 255, epochs=10):
    model.is_train = False
    model = model.to(device)
    meter = utils.AverageMeter()
    for x, y, _ in testloader:
        adv_x = x.clone().detach()
        adv_x = attack_fun(
            adv_x, y, model.backbone, epochs=epochs, eps=eps, device=args.device
        )
        x, y, adv_x = (
            x.to(device),
            y.to(device).view(
                -1,
            ),
            adv_x.to(device),
        )

        with torch.no_grad():
            out = model(adv_x)
            pred = torch.argmax(out, dim=1)
            meter.update(pred, y)
    return meter.avg


def train(args, attack_fun, model, optimizer, trainloader):
    model.is_train = True
    device = args.device
    model = model.to(device)
    for batch_idx, (x, y, _) in tqdm(
        enumerate(trainloader), total=len(trainloader), desc="Training"
    ):
        adv_x = x.clone().detach()
        adv_x = attack_fun(
            adv_x, y, model.backbone, epochs=10, eps=16 / 255, device=args.device
        )
        model.set_input(adv_x, x, y)
        model.optimize_parameters()


def main(args):
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    now = datetime.datetime.now()
    log_filename = now.strftime("%Y%m%d_%H%M%S.log")
    log_path = os.path.join(args.log_dir, log_filename)
    logger = utils.get_logger(args, parser, log_path)

    # Checkpoint directory
    model_save_name = (
        f"{args.model_name}_{args.task}_{args.base_model}_{args.at_mode}_{args.attack}"
    )
    ckpt_file = os.path.join(args.ckpt_dir, args.model_name, model_save_name)
    os.makedirs(os.path.split(ckpt_file)[0], exist_ok=True)

    # Data Loaders
    train_dataset = SARFolder(args.data_root, args.task, is_train=True)
    test_dataset = SARFolder(args.data_root, args.task, is_train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialization Model and Optimizer
    net = get_model(
        args.model_name,
        ndf=args.ndf,
        ngf=args.ngf,
        lr=args.lr,
        task=args.task,
        mtype=args.base_model,
        device=args.device,
        l1_lambda=args.l1_lambda,
        Dfeat_lambda=args.Dfeat_lambda,
    ).to(args.device)
    schedulerG = torch.optim.lr_scheduler.StepLR(
        net.optimizer_G, step_size=args.step, gamma=args.gamma, last_epoch=-1
    )
    schedulerD = torch.optim.lr_scheduler.StepLR(
        net.optimizer_D, step_size=args.step, gamma=args.gamma, last_epoch=-1
    )
    schedulerC = torch.optim.lr_scheduler.StepLR(
        net.optimizer_C, step_size=args.step, gamma=args.gamma, last_epoch=-1
    )
    try:
        backbone_dict = torch.load(
            os.path.join("./checkpoints/standard", f"standard_{args.task}_{args.base_model}_{args.at_mode}_{args.attack}.pt"),
            weights_only=True,
        )
    except Exception:
        raise Exception("Please train a standard classifier first!")

    net.backbone.load_state_dict(backbone_dict)
    for param in net.backbone.parameters():
        param.requires_grad = False

    # Training and Evaluation
    attack_func = attack_box[args.attack]
    for epoch in range(args.epochs):
        train(args, attack_func, net, None, train_loader)

        if (epoch + 1) % 2 == 0 or (epoch + 1) == args.epochs:
            torch.save(net.state_dict(), f"{ckpt_file}_epoch{epoch + 1}.pth")
            test_acc = test(
                net, test_loader, attack_func, args.device, eps=16 / 255, epochs=10
            )
            logger.info(
                f"Epoch [{epoch + 1}/{args.epochs}] Test Accuracy: {test_acc:.2f}%"
            )
        schedulerD.step()
        schedulerC.step()
        schedulerG.step()


if __name__ == "__main__":
    main(args)
