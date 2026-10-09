import argparse
import json
import os

import torch

from main import build_parser, set_seed
from ReportsNet_Final import RadialSynergyNet
from train import generate_and_save_classification_map
from utils.dataset import make_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save_dir", default=None)
    options = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = options.device
    with open(os.path.join(options.run_dir, "config.json"), encoding="utf-8") as stream:
        config = json.load(stream)
    args = build_parser().parse_args([])
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    if options.dataset_dir is not None:
        args.dataset_dir = options.dataset_dir
    set_seed(args.seed)
    train_loader, _, _ = make_dataloader(args, train_ratio=args.train_ratio)
    hsi, lidar, _ = train_loader.dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadialSynergyNet(
        hsi_channels=hsi.shape[0], lidar_channels=lidar.shape[0],
        patch_size=args.patch_size, num_classes=args.num_classes,
        num_layers=args.num_layers, K=args.radial_lines_K,
    ).to(device)
    checkpoint = os.path.join(options.run_dir, "model_best.pth")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    save_dir = options.save_dir or options.run_dir
    os.makedirs(save_dir, exist_ok=True)
    generate_and_save_classification_map(model, args, device, save_dir)


if __name__ == "__main__":
    main()
