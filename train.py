import os
import math
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix, precision_score, f1_score

import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging
from datetime import datetime
from pathlib import Path
from utils.dataset import Houston2018LazyPatchDataset


def _ensure_writable_dir(path):

    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        test_file = p / ".write_test_12345"
        test_file.touch()
        test_file.unlink()
        return str(p.resolve())
    except (PermissionError, OSError) as e:
        print(f"[Warning] Cannot write to '{path}' ({e}). Falling back to home directory.")
        fallback = Path.home() / "radial-flow-results"
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback.resolve())


def setup_logger(log_dir):

    safe_log_dir = _ensure_writable_dir(log_dir)

    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(safe_log_dir, f'{timestamp}_training.log')

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


def train(args, model, optimizer, criterion, train_loader, val_loader, device, scheduler=None, skip_map=False, w_depth=0.1, w_center=0.1):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


    base_dir = os.path.join(args.saving_path, 'radial-flow', args.dataset_name, timestamp)


    logger, log_file = setup_logger(base_dir)


    num_classes = getattr(args, 'num_classes', 0)
    if num_classes == 0:

        if args.dataset_name == 'Houston2018': num_classes = 20
        elif args.dataset_name == 'Trento': num_classes = 6
        elif args.dataset_name == 'MUUFL': num_classes = 11
        else: num_classes = 20
    else:

        logger.info(f"Using dynamic num_classes: {num_classes} (Original labels were mapped)")


    logger.info("-" * 20 + " RUN CONFIG " + "-" * 20)
    args_dict = vars(args)
    for k, v in args_dict.items():
        logger.info(f"  {k}: {v}")
    logger.info("-" * 52)

    logger.info("=" * 60)
    logger.info("Training Started")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Total epochs: {args.epoch}")
    logger.info(f"Test frequency: every {args.test_freq} epochs")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)

    best_acc = 0.


    history = {
        'train_loss': [],
        'test_loss': [],
        'train_oa': [],
        'test_oa': [],
        'epochs': []
    }

    for epoch in range(args.epoch):
        start_time = time.time()
        model.train()
        losses = AverageMeter()
        tar = np.array([])
        pre = np.array([])

        train_pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}/{args.epoch}',
                         leave=False, ncols=100)
        for batch_idx, (hsi, lidar, batch_target) in enumerate(train_pbar):
            hsi, lidar, batch_target = hsi.to(device), lidar.to(device), batch_target.to(device)


            outputs = model(hsi, lidar)

            logits_syn, aux_logits, logits_center = outputs
            loss = criterion(logits_syn, batch_target)
            for aux in aux_logits:
                loss += w_depth * criterion(aux, batch_target)
            loss += w_center * criterion(logits_center, batch_target)
            batch_out = logits_syn

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()


            if scheduler is not None:
                scheduler.step()

            losses.update(loss.data, batch_target.shape[0])


            if batch_idx % 50 == 0:
                try:
                    unique_vals, cnts = torch.unique(batch_target, return_counts=True)
                    dist = dict(zip([int(x) for x in unique_vals.tolist()], cnts.tolist()))
                    if logger:
                        logger.debug(f"Batch {batch_idx} label distribution: {dist}")
                    else:
                        print(f"Batch {batch_idx} label distribution: {dist}")
                except Exception:
                    pass


            current_logits = batch_out[0] if isinstance(batch_out, (tuple, list)) else batch_out

            batch_pred = np.argmax(current_logits.detach().cpu().numpy(), axis=1)
            batch_target_val = batch_target.detach().cpu().numpy()
            tar = np.append(tar, batch_pred)
            pre = np.append(pre, batch_target_val)

            train_pbar.set_postfix({'Loss': f'{losses.avg:.4f}'})

        end_time = time.time()
        training_time = end_time - start_time

        if (epoch % args.test_freq == 0) or (epoch == args.epoch - 1):
            try:
                train_results = compute_metrics(tar, pre, epoch=epoch, logger=logger, num_classes=num_classes) if (tar.size and pre.size) else {'OA': 0}
                train_oa = train_results.get('OA', 0)
            except Exception:
                train_oa = 0


            val_loss, results = validation(model, criterion, val_loader, device, epoch=epoch, logger=logger, num_classes=num_classes)
            aa, oa, kappa = results['AA'], results['OA'], results['Kappa']
            is_best = oa >= best_acc
            best_acc = max(oa, best_acc)


            save_checkpoint(model, is_best, base_dir,
               args=args, device=device, epoch=epoch, acc=best_acc, logger=logger, skip_map=skip_map)

            logger.info(f'Epoch: {epoch:3d} | Time: {training_time:6.2f}s | '
                       f'Train OA: {train_oa:.4f} | Val OA: {oa:.4f} | '
                       f'AA: {aa:.4f} | Kappa: {kappa:.4f} | Best: {best_acc:.4f}')
            logger.info(f'Best: {best_acc:.4f}')


            try:
                train_loss_val = float(losses.avg)
            except:
                train_loss_val = 0.0
            history['train_loss'].append(train_loss_val)

            try:
                test_loss_val = float(val_loss)
            except:
                test_loss_val = 0.0
            history['test_loss'].append(test_loss_val)

            history['train_oa'].append(float(train_oa))
            history['test_oa'].append(float(oa))
            history['epochs'].append(epoch)


            try:
                plot_training_history(history, base_dir, logger)
            except Exception as e:
                logger.warning(f"Failed to update training history plot: {e}")


    plot_training_history(history, base_dir, logger)
    logger.info("=" * 60)
    logger.info(f"Training completed. Best Val OA: {best_acc:.4f}")
    return best_acc, base_dir


def validation(model, criterion, test_loader, device, epoch=None, logger=None, num_classes=None):
    model.eval()
    with torch.no_grad():
        losses = AverageMeter()
        all_preds = np.array([])
        all_targets = np.array([])

        val_pbar = tqdm(test_loader, desc=f'Validation {epoch}', leave=False, ncols=100)
        for batch_idx, (hsi, lidar, batch_target) in enumerate(val_pbar):
            hsi, lidar, batch_target = hsi.to(device), lidar.to(device), batch_target.to(device)
            batch_out = model(hsi, lidar)


            if isinstance(batch_out, tuple):
                batch_out = batch_out[0]

            loss = criterion(batch_out, batch_target)
            losses.update(loss.data, batch_target.shape[0])


            batch_pred = np.argmax(batch_out.detach().cpu().numpy(), axis=1)
            batch_target_np = batch_target.detach().cpu().numpy()

            all_preds = np.append(all_preds, batch_pred)
            all_targets = np.append(all_targets, batch_target_np)

            val_pbar.set_postfix({'Loss': f'{losses.avg:.4f}'})


        try:
            unique_p, cnt_p = np.unique(all_preds, return_counts=True)
            unique_t, cnt_t = np.unique(all_targets, return_counts=True)
            if logger:
                logger.info(f"Validation prediction distribution: {dict(zip(unique_p.tolist(), cnt_p.tolist()))}")
                logger.info(f"Validation target distribution: {dict(zip(unique_t.tolist(), cnt_t.tolist()))}")
            else:
                print(f"Validation prediction distribution: {dict(zip(unique_p.tolist(), cnt_p.tolist()))}")
                print(f"Validation target distribution: {dict(zip(unique_t.tolist(), cnt_t.tolist()))}")
        except Exception:
            pass

        total_loss = losses.avg

        results = compute_metrics(all_preds, all_targets, epoch=epoch, logger=logger, num_classes=num_classes)

    return total_loss, results


def save_checkpoint(network, is_best, save_dir, **kwargs):
    safe_save_dir = _ensure_writable_dir(save_dir)
    logger = kwargs.get('logger', None)
    skip_map = kwargs.get('skip_map', False)


    if hasattr(network, 'module'):
        state_dict = network.module.state_dict()
    else:
        state_dict = network.state_dict()

    if is_best:
        msg = "Epoch {}: New best validation OA = {:.4f}".format(kwargs.get('epoch', ''), kwargs.get('acc', 0))
        if logger:
            logger.info(msg)
        else:
            tqdm.write(msg)


        best_path = os.path.join(safe_save_dir, 'model_best.pth')
        torch.save(state_dict, best_path)


        args = kwargs.get('args')
        if args:
            try:
                import json

                args_dict = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, list, dict))}
                with open(os.path.join(safe_save_dir, 'config.json'), 'w') as f:
                    json.dump(args_dict, f, indent=4)
            except:
                pass


        if not skip_map:
            try:
                device = kwargs.get('device')
                if args is not None and device is not None:
                    generate_and_save_classification_map(
                        network, args, device, safe_save_dir,
                        epoch=kwargs.get('epoch'), acc=kwargs.get('acc'), logger=logger
                    )
            except Exception as e:
                import traceback
                traceback.print_exc()
                warning_msg = f"Warning: Failed to generate classification map: {e}"
                (logger.warning(warning_msg) if logger else tqdm.write(warning_msg))

    elif kwargs.get('epoch', 0) % 10 == 0:
        ckpt_path = os.path.join(safe_save_dir, 'model.pth')
        torch.save(state_dict, ckpt_path)
        if logger:
            logger.info(f"Saved periodic checkpoint: {ckpt_path}")


_DATA_CACHE = {}

def generate_and_save_classification_map(model, args, device, save_dir, epoch=None, acc=None, logger=None):


    model_name = type(model).__name__
    if hasattr(model, 'module'):
        model_name = type(model.module).__name__
        model = model.module
    model.eval()

    if logger:
        logger.info(f"🗺️  Generating Aligned Map with Model: [{model_name}] (Epoch: {epoch})")


    if args.dataset_name == 'Houston2018':
        mat_path = getattr(args, 'houston2018_mat_path', os.path.join(args.dataset_dir, args.dataset_name, "Houston2018.mat"))
        if not os.path.exists(mat_path):
            if logger:
                logger.error(f"File not found: {mat_path}")
            return

        import h5py
        from torch.utils.data import DataLoader

        with h5py.File(mat_path, 'r') as m:
            idx_i = np.asarray(m[args.houston2018_indexi_key]).flatten().astype(int)
            idx_j = np.asarray(m[args.houston2018_indexj_key]).flatten().astype(int)
            raw_gt = np.asarray(m[args.houston2018_gt_key]).flatten().astype(int)

        if np.min(idx_i) >= 1:
            idx_i -= 1
        if np.min(idx_j) >= 1:
            idx_j -= 1

        H_map, W_map = np.max(idx_i) + 1, np.max(idx_j) + 1
        pred_map = np.zeros((H_map, W_map), dtype=np.int32)
        gt_map = np.zeros((H_map, W_map), dtype=np.int32)

        valid_orig_labels = getattr(args, 'original_labels', [])
        if valid_orig_labels:
            valid_mask = np.isin(raw_gt, valid_orig_labels)
            gt_map[idx_i, idx_j] = np.where(valid_mask, raw_gt, 0)
            valid_indices = np.where(valid_mask)[0]
        else:
            gt_map[idx_i, idx_j] = raw_gt
            valid_indices = np.where(raw_gt > 0)[0]

        if logger:
            logger.info(f"Inferring Houston2018 valid samples lazily: {len(valid_indices)}/{len(raw_gt)}")

        map_dataset = Houston2018LazyPatchDataset(
            mat_path,
            args.houston2018_hsi_key,
            args.houston2018_lidar_key,
            raw_gt,
            valid_indices,
            args.houston2018_hsi_sample_axis,
            args.houston2018_hsi_channel_axis,
            args.houston2018_lidar_sample_axis,
            args.houston2018_lidar_channel_axis,
            args.scaler_hsi,
            args.scaler_lidar,
            train=False,
            label_map=args.label_map,
            return_indices=True
        )
        batch_size = getattr(args, 'val_batch_size', 1024)
        map_loader = DataLoader(map_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        reverse_map = getattr(args, 'reverse_label_map', None)

        with torch.no_grad():
            pbar = tqdm(map_loader, desc="Inferring Houston2018", leave=False)
            for h_batch, l_batch, _, sample_idx in pbar:
                h_batch = h_batch.to(device)
                l_batch = l_batch.to(device)
                out = model(h_batch, l_batch)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                preds = torch.argmax(out, dim=1).cpu().numpy()
                sample_idx = sample_idx.numpy()

                current_i = idx_i[sample_idx]
                current_j = idx_j[sample_idx]
                if reverse_map:
                    mapped_preds = np.array([reverse_map.get(int(p), int(p) + 1) for p in preds])
                    pred_map[current_i, current_j] = mapped_preds
                else:
                    pred_map[current_i, current_j] = preds + 1


        num_classes = 20


        vis_max_label = num_classes


        gt = gt_map

    else:


        if logger:
            logger.info(f"Generating classification map for epoch {epoch}...")


        hsi_path = os.path.join(args.dataset_dir, args.dataset_name, "HSI.mat")
        import scipy.io as sio
        import h5py
        try:
            m = sio.loadmat(hsi_path)
            h_keys = [k for k in m.keys() if 'hsi' in k.lower() or 'data' in k.lower()]
            if not h_keys:
                raise KeyError("No HSI key found")
            hsi = m[h_keys[0]].astype(np.float32)
        except Exception as e:
            if logger: logger.error(f"Failed to load HSI via loadmat: {e}")
            with h5py.File(hsi_path, 'r') as f:
                h_keys = [k for k in f.keys() if 'hsi' in k.lower() or 'data' in k.lower()]
                hsi = np.array(f[h_keys[0]]).astype(np.float32)


        gt_path = os.path.join(args.dataset_dir, args.dataset_name, "gt.mat")
        try:

             m_gt = sio.loadmat(gt_path)
             g_key = [k for k in m_gt.keys() if 'gt' in k.lower() or 'label' in k.lower() or 'y' in k.lower()][0]
             gt = m_gt[g_key].astype(np.int32)
        except:
             if logger: logger.warning("Could not find generic 'gt' key, trying hardcoded 'gt'.")
             gt = sio.loadmat(gt_path)['gt'].astype(np.int32)

        Hg, Wg = gt.shape

        if hsi.ndim == 3:
            s = hsi.shape

            if s[1] == Hg and s[2] == Wg:
                hsi = hsi.transpose(1, 2, 0)

            elif s[0] == Hg and s[1] == Wg:
                pass

            elif s[0] == Wg and s[1] == Hg:
                hsi = hsi.transpose(1, 0, 2)

            elif s[1] == Wg and s[2] == Hg:
                hsi = hsi.transpose(2, 1, 0)
            else:
                if logger: logger.warning(f"HSI shape {s} does not match GT shape {gt.shape}. Attempting heuristic fix.")


                min_dim = np.argmin(s)
                if min_dim == 0:
                    hsi = hsi.transpose(1, 2, 0)
                elif min_dim == 1:
                    pass

        H, W, C_hsi = hsi.shape
        if logger: logger.info(f"HSI loaded: {hsi.shape}, GT: {gt.shape}")

        if H != Hg or W != Wg:
            if logger: logger.error(f"Shape Mismatch Fatal: HSI({H},{W}) vs GT({Hg},{Wg})")


        hsi_flat = hsi.reshape(-1, C_hsi)


        if hasattr(args, 'scaler_hsi') and args.scaler_hsi is not None:

            if args.scaler_hsi.n_features_in_ == C_hsi:
                hsi_flat = args.scaler_hsi.transform(hsi_flat)
            else:
                if logger: logger.warning(f"Scaler dim mismatch ({args.scaler_hsi.n_features_in_} vs {C_hsi}). Re-fitting scaler.")
                from sklearn.preprocessing import MinMaxScaler
                scaler_hsi = MinMaxScaler()
                hsi_flat = scaler_hsi.fit_transform(hsi_flat)
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler_hsi = MinMaxScaler()
            hsi_flat = scaler_hsi.fit_transform(hsi_flat)


        if args.use_pca:
            if hasattr(args, 'pca_model_hsi') and args.pca_model_hsi is not None:
                if args.pca_model_hsi.n_features_in_ == hsi_flat.shape[1]:
                     hsi_flat = args.pca_model_hsi.transform(hsi_flat)
                else:
                     if logger: logger.warning("PCA dimension mismatch. Skipping PCA (This will likely fail model execution).")
            else:
                from sklearn.decomposition import PCA


                pca_model = PCA(n_components=args.pca_component, random_state=42)
                hsi_flat = pca_model.fit_transform(hsi_flat)

        hsi = hsi_flat.reshape(H, W, -1)


        if args.dataset_name in ['Augsburg', 'Berlin']:
            lidar_path = os.path.join(args.dataset_dir, args.dataset_name, "SAR.mat")
            key_candidates = ['SAR', 'sar']
        else:
            lidar_path = os.path.join(args.dataset_dir, args.dataset_name, "LiDAR.mat")
            key_candidates = ['LiDAR', 'lidar', 'dsm', 'DSM']

        try:
            m_l = sio.loadmat(lidar_path)
            l_key = next((k for k in m_l.keys() if k in key_candidates), None)
            if l_key is None:
                l_key = [k for k in m_l.keys() if not k.startswith('__')][0]
            lidar = m_l[l_key].astype(np.float32)
        except Exception:

            if logger: logger.warning("Failed to load LiDAR, using zeros.")
            lidar = np.zeros((H, W), dtype=np.float32)


        if lidar.ndim == 2:
            pass
        elif lidar.ndim == 3:

             if lidar.shape[1] == H and lidar.shape[2] == W:
                 lidar = lidar.transpose(1, 2, 0)

        if lidar.shape[0] != H or lidar.shape[1] != W:

             if lidar.shape[0] == W and lidar.shape[1] == H:
                  lidar = lidar.transpose(1, 0)

        if lidar.ndim == 2:
            lidar = np.expand_dims(lidar, axis=2)

        H_l, W_l, C_l = lidar.shape
        lidar_flat = lidar.reshape(-1, C_l)


        if hasattr(args, 'scaler_lidar') and args.scaler_lidar is not None:
             if args.scaler_lidar.n_features_in_ == C_l:
                 lidar_flat = args.scaler_lidar.transform(lidar_flat)
             else:
                 from sklearn.preprocessing import MinMaxScaler
                 lidar_flat = MinMaxScaler().fit_transform(lidar_flat)
        else:
            from sklearn.preprocessing import MinMaxScaler
            lidar_flat = MinMaxScaler().fit_transform(lidar_flat)
        lidar = lidar_flat.reshape(H, W, -1)


        valid_orig_labels = getattr(args, 'original_labels', [])
        if valid_orig_labels:

             mask_valid = np.isin(gt, valid_orig_labels)
             gt[~mask_valid] = 0

        pad = args.patch_size // 2
        modes = ['symmetric', 'reflect']
        hsi_p = np.pad(hsi, ((pad, pad), (pad, pad), (0, 0)), mode=modes[args.patch_size % 2])
        lidar_p = np.pad(lidar, ((pad, pad), (pad, pad), (0, 0)), mode=modes[args.patch_size % 2])

        pred_map = np.zeros((H, W), dtype=np.int32)

        batch_coords = []
        batch_hsi = []
        batch_lidar = []
        batch_size = getattr(args, 'val_batch_size', 512)

        valid_idx = np.argwhere(gt > 0)
        total_pixels = len(valid_idx)

        with torch.no_grad():
            for idx, (i, j) in enumerate(valid_idx):
                h_patch = hsi_p[i: i + args.patch_size, j: j + args.patch_size]
                l_patch = lidar_p[i: i + args.patch_size, j: j + args.patch_size]
                batch_coords.append((i, j))
                batch_hsi.append(h_patch)
                batch_lidar.append(l_patch)

                if len(batch_coords) >= batch_size or idx == len(valid_idx) - 1:
                    h_tensor = torch.tensor(np.array(batch_hsi), dtype=torch.float32).permute(0, 3, 1, 2).contiguous().to(device)
                    l_tensor = torch.tensor(np.array(batch_lidar), dtype=torch.float32).permute(0, 3, 1, 2).contiguous().to(device)
                    out = model(h_tensor, l_tensor)


                    if isinstance(out, (tuple, list)):
                        out = out[0]

                    raw_preds = np.argmax(out.detach().cpu().numpy(), axis=1)


                    reverse_map = getattr(args, 'reverse_label_map', None)
                    for (coords_pair, p_idx) in zip(batch_coords, raw_preds):
                        x, y = coords_pair

                        orig_label = reverse_map.get(int(p_idx), int(p_idx) + 1) if reverse_map else int(p_idx) + 1
                        pred_map[x, y] = orig_label

                    batch_coords = []
                    batch_hsi = []
                    batch_lidar = []

                    if logger and idx % (batch_size * 5) == 0:
                        progress = (idx / total_pixels) * 100


        if logger:
             unique, counts = np.unique(pred_map, return_counts=True)
             stats_str = ", ".join([f"Cls{k}:{v}" for k,v in zip(unique, counts) if k!=0])
             logger.info(f"    Prediction Stats: {stats_str}")


    label_map = getattr(args, 'label_map', None)
    reverse_label_map = getattr(args, 'reverse_label_map', None)
    original_labels = getattr(args, 'original_labels', [])
    num_classes = getattr(args, 'num_classes', len(original_labels) if original_labels else 0)


    use_orig_label = True


    if args.dataset_name == 'Trento':

        NEW_LABEL_COLORS = [
            '#3D26A8',
            '#4659F8',
            '#2796EB',
            '#80C9BC',
            '#80CB58',
            '#FCBB3D',
            '#F9FA14',
        ]
        vis_max_label = 6
    elif args.dataset_name == 'MUUFL':

        NEW_LABEL_COLORS = [
            '#3D26A8',
            '#463FE3',
            '#455FFB',
            '#2E81F9',
            '#239FE4',
            '#02B6CC',
            '#2DC4A4',
            '#63CC6E',
            '#B8C431',
            '#F4B93A',
            '#FFDA1D',
            '#F9FA14',
        ]
        vis_max_label = 11
    elif args.dataset_name == 'Houston2018':

        NEW_LABEL_COLORS = [
            '#3D26A8',
            '#463CDF',
            '#4659F8',
            '#808080',
            '#337AFC',
            '#2796EB',
            '#808080',
            '#808080',
            '#17AEDA',
            '#13BEB8',
            '#3DE68B',
            '#8BE64B',
            '#808080',
            '#CAC028',
            '#808080',
            '#808080',
            '#FCBB3D',
            '#808080',
            '#FFF319',
            '#808080',
            '#F9FA14',
        ]
        vis_max_label = 20
    else:

        max_orig = max(original_labels) if original_labels else num_classes
        vis_max_label = max_orig
        NEW_LABEL_COLORS = ['#000000'] + [plt.cm.tab20(i % 20) for i in range(max_orig)]


    color_list = NEW_LABEL_COLORS


    full_color_list = color_list[:vis_max_label + 1]
    if len(full_color_list) < vis_max_label + 1:
        full_color_list += ['#808080'] * (vis_max_label + 1 - len(full_color_list))

    print(f"[Color Scheme] Dataset: {args.dataset_name}")
    print(f"[Color Mapping] 原始标签直接对应颜色:")
    print(f"  标签 0 (背景) -> 颜色: {color_list[0]}")
    for i in range(1, min(vis_max_label + 1, len(color_list))):
        if color_list[i] != '#808080':
            print(f"  标签 {i} -> 颜色: {color_list[i]}")

    cmap = ListedColormap(full_color_list[:vis_max_label + 1])


    gt_display = gt.copy()
    pred_display = pred_map.copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im0 = axes[0].imshow(pred_display, cmap=cmap, vmin=0, vmax=vis_max_label, interpolation='nearest')
    axes[0].set_title('Prediction')

    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, ticks=range(vis_max_label + 1))
    cbar0.ax.set_yticklabels([str(i) for i in range(vis_max_label + 1)])

    im1 = axes[1].imshow(gt_display, cmap=cmap, vmin=0, vmax=vis_max_label, interpolation='nearest')
    axes[1].set_title('Ground Truth')

    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, ticks=range(vis_max_label + 1))
    cbar1.ax.set_yticklabels([str(i) for i in range(vis_max_label + 1)])

    title = f"Epoch {epoch} OA={acc:.4f}" if (epoch is not None and acc is not None) else "Classification Map"
    plt.suptitle(title)

    fname = os.path.join(save_dir, f'class_map_epoch{epoch}_oa{acc:.4f}.png') if (epoch is not None and acc is not None) else os.path.join(save_dir, 'class_map.png')
    plt.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)

    if logger:
        logger.info(f"Classification map saved: {fname}")


    fig_pure = plt.figure(figsize=(10, 10), frameon=False)
    ax_pure = plt.Axes(fig_pure, [0., 0., 1., 1.])
    ax_pure.set_axis_off()
    fig_pure.add_axes(ax_pure)
    ax_pure.imshow(pred_display, cmap=cmap, vmin=0, vmax=vis_max_label, interpolation='nearest')
    fname_pure_pred = os.path.join(save_dir, f'pred_pure_epoch{epoch}_oa{acc:.4f}.png') if (epoch is not None and acc is not None) else os.path.join(save_dir, 'pred_pure.png')
    fig_pure.savefig(fname_pure_pred, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig_pure)


    fig_pure_gt = plt.figure(figsize=(10, 10), frameon=False)
    ax_pure_gt = plt.Axes(fig_pure_gt, [0., 0., 1., 1.])
    ax_pure_gt.set_axis_off()
    fig_pure_gt.add_axes(ax_pure_gt)
    ax_pure_gt.imshow(gt_display, cmap=cmap, vmin=0, vmax=vis_max_label, interpolation='nearest')
    fname_pure_gt = os.path.join(save_dir, f'gt_pure_epoch{epoch}_oa{acc:.4f}.png') if (epoch is not None and acc is not None) else os.path.join(save_dir, 'gt_pure.png')
    fig_pure_gt.savefig(fname_pure_gt, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig_pure_gt)

    if logger:
        logger.info(f"Pure classification maps saved: {fname_pure_pred}, {fname_pure_gt}")


def compute_metrics(pred, target, epoch=None, logger=None, num_classes=None):

    results = {}
    labels = range(num_classes) if num_classes is not None else None
    cm = confusion_matrix(target, pred, labels=labels)
    results['Confusion matrix'] = cm

    oa = 1. * np.trace(cm) / np.sum(cm)
    results['OA'] = oa

    n_classes = cm.shape[0]
    pa = np.array([1. * cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else np.nan for i in range(n_classes)])
    results['PA'] = pa

    epoch_str = f"Epoch {epoch}" if epoch is not None else "Metrics"
    print(f"\n[{epoch_str}] Per-class accuracy:")
    for i, acc in enumerate(pa):
        class_acc_msg = f"  Class {i}: {acc:.4f}"
        print(class_acc_msg)
        if logger:
            logger.info(class_acc_msg)

    aa = np.mean(pa)
    results['AA'] = aa

    precision = precision_score(target, pred, average='macro', zero_division=0)
    f1 = f1_score(target, pred, average='macro', zero_division=0)
    results['Precision'] = precision
    results['F1'] = f1

    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(np.sum(cm) * np.sum(cm))
    kappa = (oa - pe) / (1 - pe)
    results['Kappa'] = kappa

    if logger:
        summary_msg = f"{epoch_str} Summary: OA={oa:.4f}, AA={aa:.4f}, Kappa={kappa:.4f}, PR={precision:.4f}, F1={f1:.4f}"
        logger.info(summary_msg)

    return results


class AverageMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def step_learning_rate(args, epoch, optimizer):
    total_epochs = args.epoch
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = 5.
    elif epoch < int(0.4 * total_epochs):
        lr_adj = 1.
    elif epoch < int(0.7 * total_epochs):
        lr_adj = 0.1
    elif epoch < int(1 * total_epochs):
        lr_adj = 0.01
    else:
        lr_adj = 0.

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj


def cosine_learning_rate(args, epoch, optimizer):
    total_epochs = args.epoch
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = 1.
    else:
        lr_adj = 1 / 2 * (1 + math.cos(math.pi * epoch / (total_epochs - warm_epochs)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj


def plot_training_history(history, save_dir, logger=None):


    safe_save_dir = _ensure_writable_dir(save_dir)
    epochs = history['epochs']

    plt.figure(figsize=(12, 5))


    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', lw=2)
    plt.plot(epochs, history['test_loss'], label='Test Loss', color='red', lw=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)


    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_oa'], label='Train OA', color='green', lw=2)
    plt.plot(epochs, history['test_oa'], label='Test OA', color='orange', lw=2)
    plt.title('Training and Validation OA')
    plt.xlabel('Epochs')
    plt.ylabel('Overall Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(safe_save_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()

    if logger:
        logger.info(f"Training history curves saved to: {plot_path}")
    else:
        print(f"Training history curves saved to: {plot_path}")
