import argparse
import math
import os
import random

import numpy as np
import torch

from ReportsNet_Final import RadialSynergyNet
from train import generate_and_save_classification_map, train, validation
from utils.dataset import make_dataloader


def build_parser():
    parser = argparse.ArgumentParser(description="Radial-Flow hyperspectral/LiDAR classification")
    parser.add_argument("--device", default="0")
    parser.add_argument("--dataset_name", default="Houston2013")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--patch_size", type=int, default=11)
    parser.add_argument("--use_pca", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--pca_component", type=int, default=30)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--scheduler_type", choices=("Step", "OneCycle", "WarmupCosine"), default="WarmupCosine")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--split_seed", type=int, default=999)
    parser.add_argument("--saving_path", default="./runs")
    parser.add_argument("--test_freq", type=int, default=1)
    parser.add_argument("--is_train", type=parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--lidar_channels", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--radial_lines_K", type=int, default=12)
    parser.add_argument("--balanced_sampling", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_rotation_aug", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--w_depth", type=float, default=0.1)
    parser.add_argument("--w_center", type=float, default=0.1)
    return parser


def parse_bool(value):
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected True or False")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_scheduler(args, optimizer, train_loader):
    if args.scheduler_type == "OneCycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
            epochs=args.epoch, pct_start=0.1, anneal_strategy="cos", final_div_factor=1000
        )
    if args.scheduler_type == "WarmupCosine":
        warmup = args.warmup_epochs * max(1, len(train_loader))
        total = args.epoch * max(1, len(train_loader))
        def schedule(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20 * max(1, len(train_loader)), gamma=0.5)


def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = make_dataloader(args, train_ratio=args.train_ratio)
    sample_hsi, sample_lidar, _ = train_loader.dataset[0]
    in_channels = int(sample_hsi.shape[0])
    args.lidar_channels = int(sample_lidar.shape[0])
    num_classes = int(args.num_classes)

    model = RadialSynergyNet(
        hsi_channels=in_channels, lidar_channels=args.lidar_channels,
        hidden_dim=64, patch_size=args.patch_size, num_classes=num_classes,
        num_layers=args.num_layers, K=args.radial_lines_K,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(args, optimizer, train_loader)
    criterion = torch.nn.CrossEntropyLoss()

    if args.is_train:
        _, save_dir = train(
            args, model, optimizer, criterion, train_loader, val_loader, device,
            scheduler=scheduler, skip_map=True,
            w_depth=args.w_depth, w_center=args.w_center,
        )
        checkpoint = os.path.join(save_dir, "model_best.pth")
    else:
        save_dir = args.saving_path
        checkpoint = os.path.join(save_dir, "model_best.pth")

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError("Radial-Flow checkpoint not found: " + checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, results = validation(model, criterion, test_loader, device, num_classes=num_classes)
    print("Test Results - OA: {:.4f}, AA: {:.4f}, Kappa: {:.4f}".format(
        results["OA"], results["AA"], results["Kappa"]))
    generate_and_save_classification_map(model, args, device, save_dir, epoch="Final", acc=results["OA"])


if __name__ == "__main__":
    main()
