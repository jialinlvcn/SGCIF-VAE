import argparse
import utils
import os
import torch
from model import get_model
from torch.utils.data import DataLoader
from dataset.sar_dataset import SARFolder

parser = argparse.ArgumentParser(description="SGCIF-VAE Training")
parser.add_argument("--data_root", default="./sar_datasets", type=str)
parser.add_argument(
    "--model_name", default="sgcif", type=str, help="Model name, e.g. sgcif"
)
parser.add_argument("--task", default="SAR_ACD", type=str)
parser.add_argument("--seed", default=666, type=int, help="seed")
parser.add_argument("--log_dir", default="./logs/testing", type=str)
parser.add_argument("--ckpt_path", default="./checkpoints/best-at-resnet-ACD-pgd.pt", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--dim", default=2048, type=int)
parser.add_argument("--fdim", default=128, type=int)
parser.add_argument("--base_model", default="resnet", type=str)
parser.add_argument("--attack", default="pgd", type=str)
parser.add_argument("--at_mode", default="at", type=str)
args = parser.parse_args()

utils.seed_everything(args.seed)


def main(args):
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_filename = (
        f"{args.model_name}_{args.task}_{args.base_model}_{args.at_mode}_{args.attack}"
    )
    log_path = os.path.join(args.log_dir, log_filename)

    # Load Model with Checkpoint
    net = get_model(
        model_name=args.model_name,
        d=args.fdim,
        z=args.dim,
        with_classifier=True,
        mtype=args.base_model,
        task=args.task,
    ).to(args.device)
    model_dict = torch.load(args.ckpt_path, weights_only=True)
    net.load_state_dict(model_dict)

    # Prepare Attack Function and DataLoader
    test_dataset = SARFolder(args.data_root, args.task, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Evaluate Model with ALL Attack
    eval_result = utils.norm_eval(net, test_loader, log_path, args.device)
    for key, value in zip(eval_result[0], eval_result[1]):
        print(f"{key} : {value} %")

if __name__ == "__main__":
    main(args)
