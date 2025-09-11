import os
import argparse
import datetime
import utils
import torch
from torch.utils.data import DataLoader
from dataset.sar_dataset import SARFolder
from model import get_model
from attack import attack_box
from loss_fns import CIFLoss

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
    "--model_name", default="sgcif", type=str, help="Model name, e.g. sgcif"
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
parser.add_argument(
    "--dim", default=2048, type=int, help="Feature dimension of main model"
)
parser.add_argument(
    "--fdim", default=128, type=int, help="Feature dimension of classifier"
)
parser.add_argument(
    "--step", default=50, type=int, help="Steps interval for logging or saving model"
)
parser.add_argument("--lr", default=5e-4, type=float, help="Learning rate")
parser.add_argument(
    "--alpha", default=0.1, type=float, help="Alpha coefficient in loss function"
)
parser.add_argument(
    "--beta", default=1.0, type=float, help="Beta coefficient in loss function"
)
parser.add_argument(
    "--lam", default=0.1, type=float, help="Lambda coefficient in loss function"
)
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

utils.seed_everything(args.seed)


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
        model_name=args.model_name,
        d=args.fdim,
        z=args.dim,
        with_classifier=True,
        mtype=args.base_model,
        task=args.task,
    ).to(args.device)
    optimizer = torch.optim.Adam(
        [{"params": net.parameters()}], lr=args.lr, betas=(0.5, 0.9)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step, gamma=args.gamma, last_epoch=-1
    )

    # Loss Function
    loss_function = CIFLoss(
        alpha=args.alpha, beta=args.beta, lam=args.lam, at_method=args.at_mode, dim=args.dim
    )

    # Training and Evaluation
    attack_func = attack_box[args.attack]
    for epoch in range(args.epochs):
        utils.train_one_epoch(
            net,
            train_loader,
            loss_function,
            attack_func,
            optimizer,
            args.device,
            logger,
            eps=16 / 255,
            epochs=args.epochs,
        )
        scheduler.step()

        if (epoch + 1) % 2 == 0 and epoch > 50 or (epoch + 1) == args.epochs:
            torch.save(net.state_dict(), f"{ckpt_file}_epoch{epoch+1}.pth")
            test_acc = utils.evaluate(net, test_loader, attack_func, args.device)
            logger.info(f"Epoch [{epoch+1}/{args.epochs}] Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main(args)
