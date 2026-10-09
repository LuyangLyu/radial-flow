import argparse
import os
import sys
import gc


if "--device" in sys.argv:
    idx = sys.argv.index("--device")
    if idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[idx+1]

import numpy as np
import torch
import torch.nn.functional as F
from scipy import io
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from matplotlib.colors import ListedColormap


class RobustRotationDataset(Dataset):


    def __init__(self, hsi_padded, lidar_padded, gt, mask, patch_size, angle=0, label_map=None, safe_size=0):
        self.hsi_padded = hsi_padded
        self.lidar_padded = lidar_padded
        self.gt = gt
        self.mask = mask
        self.patch_size = patch_size
        self.angle = angle
        self.label_map = label_map
        self.safe_size = safe_size

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, index):

        r, c = self.mask[index]
        label_val = self.gt[r, c]


        h_start, w_start = r, c
        h_end, w_end = r + self.safe_size, c + self.safe_size

        hsi_patch = self.hsi_padded[h_start:h_end, w_start:w_end, :]
        lidar_patch = self.lidar_padded[h_start:h_end, w_start:w_end, :]


        hsi_t = torch.from_numpy(hsi_patch).permute(2, 0, 1).float()
        lidar_t = torch.from_numpy(lidar_patch).permute(2, 0, 1).float()

        if self.angle != 0:

            hsi_t = self._rotate_tensor(hsi_t, self.angle)
            lidar_t = self._rotate_tensor(lidar_t, self.angle)


        start = (self.safe_size - self.patch_size) // 2
        hsi_t = hsi_t[:, start:start+self.patch_size, start:start+self.patch_size]
        lidar_t = lidar_t[:, start:start+self.patch_size, start:start+self.patch_size]


        if self.label_map:
            gt_val = self.label_map.get(label_val, 0)
        else:
            gt_val = label_val - 1

        return hsi_t, lidar_t, torch.tensor(gt_val).long()

    def _rotate_tensor(self, img_tensor, angle):


        angle = angle % 360
        if angle == 90:
            return torch.rot90(img_tensor, k=1, dims=[1, 2])
        elif angle == 180:
            return torch.rot90(img_tensor, k=2, dims=[1, 2])
        elif angle == 270:
            return torch.rot90(img_tensor, k=3, dims=[1, 2])
        elif angle == 0:
            return img_tensor

        img_tensor = img_tensor.unsqueeze(0)
        theta = math.radians(angle)

        rot_mat = torch.tensor([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0]
        ], dtype=torch.float32).unsqueeze(0)


        grid = F.affine_grid(rot_mat, img_tensor.size(), align_corners=True)
        rotated = F.grid_sample(img_tensor, grid, mode='bicubic', padding_mode='reflection', align_corners=True)
        return rotated.squeeze(0)


def compute_metrics(pred, target, num_classes):
    cm = confusion_matrix(target, pred, labels=range(num_classes))
    oa = 1. * np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    pa = np.array([1. * cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0 for i in range(num_classes)])
    aa = np.mean(pa)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(np.sum(cm) * np.sum(cm)) if np.sum(cm) > 0 else 1
    kappa = (oa - pe) / (1 - pe) if (1-pe) != 0 else 0
    pr = precision_score(target, pred, average='weighted', zero_division=0)
    f1 = f1_score(target, pred, average='weighted', zero_division=0)
    return oa, aa, kappa, pr, f1

def load_model(args, device, in_channels, num_classes):
    from ReportsNet_Final import RadialSynergyNet
    model = RadialSynergyNet(
        hsi_channels=in_channels, lidar_channels=args.lidar_channels,
        patch_size=args.patch_size, num_classes=num_classes,
        num_layers=args.num_layers, K=args.radial_lines_K,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    return model


def save_classification_map(pred_map, gt, save_path, title, dataset_name):
    if dataset_name == 'Trento':
        custom_colors = [
            '#352A86', '#046CE0', '#069CCF', '#37B89D', '#A5BE6A', '#FDBE3D', '#FFFF05', '#800080',
            '#228B22', '#FFC0CB', '#000000',
            '#E6194B', '#F58231', '#911EB4', '#46F0F0', '#F032E6', '#BCF60C', '#FABEBE', '#008080',
            '#E6BEFF', '#9A6324', '#FFFAC8', '#800000', '#AAFFC3', '#808000', '#FFD8B1', '#000075'
        ]
    else:
        custom_colors = [
            '#3C2875', '#21429B', '#006DB8', '#008EBC', '#0098BD',
            '#00A4A7', '#00A98A', '#7FBA68', '#CFBD4D', '#FEBF37',
            '#FFDE00',
            '#E6194B', '#F58231', '#911EB4', '#46F0F0', '#F032E6', '#BCF60C', '#FABEBE', '#008080',
            '#E6BEFF', '#9A6324', '#FFFAC8', '#800000', '#AAFFC3', '#808000', '#FFD8B1', '#000075',
            '#808080', '#FFFFFF', '#000000', '#D2691E', '#FF1493', '#00BFFF', '#32CD32', '#FF4500'
        ]

    vis_max_label = int(np.max(gt))

    color_list = custom_colors * ((vis_max_label + 1 + len(custom_colors) - 1) // len(custom_colors))
    color_list = color_list[:vis_max_label + 1]
    cmap = ListedColormap(color_list)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axes[0].imshow(pred_map, cmap=cmap, vmin=0, vmax=vis_max_label)
    axes[0].set_title('Prediction')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(gt, cmap=cmap, vmin=0, vmax=vis_max_label)
    axes[1].set_title('Ground Truth')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_oa_curve(angles, oas, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(angles, oas, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title('Inference OA vs Rotation Angle')
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('Overall Accuracy (OA)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(angles)
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--dataset_name", type=str, default="Trento")
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--device", type=str, default="3")
    parser.add_argument("--patch_size", type=int, default=11)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--angles", type=str, default="0,15,30,45,60,75,90", help="Comma separated angles")
    parser.add_argument("--save_dir", type=str, default="./rotation_test/model_default", help="Directory to save results")
    parser.add_argument("--radial_lines_K", type=int, default=10, help="Number of radial lines")
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    from utils.dataset import load_processed_dataset, select_mask
    hsi, lidar, gt = load_processed_dataset(argparse.Namespace(
        dataset_dir=args.dataset_dir, dataset_name=args.dataset_name, use_pca=False
    ))


    data_dir = os.path.join(args.dataset_dir, args.dataset_name)
    ts_path = os.path.join(data_dir, "TSLabel.mat")


    all_valid_mask = np.array(np.nonzero(gt > 0)).T


    test_mask_bool = np.zeros_like(gt, dtype=bool)
    if os.path.exists(ts_path):
        ts_gt = io.loadmat(ts_path)['TSLabel']
        test_mask_bool[ts_gt > 0] = True
    else:
        test_mask_bool[gt > 0] = True


    total_pixels = gt.size
    unique_labels_in_gt, counts = np.unique(gt, return_counts=True)
    threshold = 0.002 * total_pixels
    unique_labels = [l for l, c in zip(unique_labels_in_gt, counts) if l > 0 and c >= threshold]
    unique_labels = sorted(unique_labels)

    num_classes = len(unique_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    reverse_label_map = {new: old for new, old in enumerate(unique_labels)}


    mask_valid = np.isin(gt, unique_labels)
    gt[~mask_valid] = 0
    all_valid_mask = np.array(np.nonzero(gt > 0)).T

    in_channels = hsi.shape[2]
    args.lidar_channels = lidar.shape[2] if lidar.ndim == 3 else 1


    patch_size = args.patch_size
    safe_size = int(math.ceil(patch_size * 1.5))
    if safe_size % 2 == 0: safe_size += 1
    pad = safe_size // 2
    hsi_padded = np.pad(hsi, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    lidar_padded = np.pad(lidar, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')


    model = load_model(args, device, in_channels, num_classes)

    angle_list = [float(a) for a in args.angles.split(',')]
    results_all = {}
    oa_history = []

    print(f"\nStarting Robustness Experiment on {args.dataset_name}")
    print(f"Testing angles: {angle_list}")

    for angle in angle_list:
        test_ds = RobustRotationDataset(hsi_padded, lidar_padded, gt, all_valid_mask, args.patch_size,
                                       angle=angle, label_map=label_map, safe_size=safe_size)
        loader = DataLoader(test_ds, batch_size=512, num_workers=4, shuffle=False)

        pred_map = np.zeros_like(gt)
        test_preds = []
        test_targets = []

        with torch.no_grad():
            for idx, (batch_h, batch_l, batch_t) in enumerate(tqdm(loader, desc=f"Angle {angle}°")):
                batch_h, batch_l = batch_h.to(device), batch_l.to(device)
                out = model(batch_h, batch_l)
                if isinstance(out, (tuple, list)): out = out[0]
                preds = torch.argmax(out, dim=1).cpu().numpy()


                curr_batch_size = batch_h.shape[0]
                start_idx = idx * 512
                for i in range(curr_batch_size):
                    r_orig, c_orig = all_valid_mask[start_idx + i]
                    p_val = preds[i]

                    orig_p_val = reverse_label_map.get(p_val, p_val + 1)
                    pred_map[r_orig, c_orig] = orig_p_val


                    if test_mask_bool[r_orig, c_orig]:
                        test_preds.append(p_val)
                        test_targets.append(batch_t[i].item())

        oa, aa, kappa, pr, f1 = compute_metrics(np.array(test_preds), np.array(test_targets), num_classes)
        results_all[angle] = {'OA': oa, 'AA': aa, 'Kappa': kappa, 'PR': pr, 'F1': f1}
        oa_history.append(oa)
        print(f">> Angle {angle}°: OA={oa:.4f}, AA={aa:.4f}, Kappa={kappa:.4f}, PR={pr:.4f}, F1={f1:.4f}")


        save_classification_map(pred_map, gt,
                               os.path.join(args.save_dir, f"map_angle_{angle}.png"),
                               f"Angle {angle}° OA={oa:.4f}", args.dataset_name)


        del loader, test_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    summary_path = os.path.join(args.save_dir, f"robustness_results.txt")
    with open(summary_path, "w") as f_out:
        header = f"{'Angle':<10} | {'OA':<10} | {'AA':<10} | {'Kappa':<10} | {'PR':<10} | {'F1':<10}"
        print("\n" + "="*70)
        print(f"Summary Results for {args.dataset_name}")
        print(header)
        print("-" * 70)
        f_out.write(f"Summary Results for {args.dataset_name}\n" + header + "\n" + "-"*70 + "\n")

        for ang in angle_list:
            res = results_all[ang]
            line = f"{ang:<10}° | {res['OA']:<10.4f} | {res['AA']:<10.4f} | {res['Kappa']:<10.4f} | {res['PR']:<10.4f} | {res['F1']:<10.4f}"
            print(line)
            f_out.write(line + "\n")
        print("="*70)
        print(f"Results saved to: {summary_path}")


    plot_oa_curve(angle_list, oa_history, os.path.join(args.save_dir, "oa_curve.png"))
    print(f"OA curve saved to: {os.path.join(args.save_dir, 'oa_curve.png')}")

if __name__ == "__main__":
    main()
