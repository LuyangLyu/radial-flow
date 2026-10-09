import os
import random
import numpy as np
import torch
from scipy import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor


class HXDataset(Dataset):

    def __init__(self, args, hsi, lidar, gt, mask, transform=ToTensor(), train=False, label_map=None):
        self.dataset_name = args.dataset_name
        modes = ['symmetric', 'reflect']
        self.train = train
        self.mask = mask
        self.pad = args.patch_size // 2
        self.patch_size = args.patch_size
        self.label_map = label_map
        self.use_rotation_aug = getattr(args, 'use_rotation_aug', False)

        self.hsi = np.pad(hsi, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[self.patch_size % 2])
        if lidar.ndim == 2:
            self.lidar = np.pad(lidar, ((self.pad, self.pad),
                                    (self.pad, self.pad)), mode=modes[self.patch_size % 2])
        elif lidar.ndim == 3:
            self.lidar = np.pad(lidar, ((self.pad, self.pad),
                                    (self.pad, self.pad), (0, 0)), mode=modes[self.patch_size % 2])
        self.gt = gt
        if transform:
            self.transform = transform

    def __getitem__(self, index):
        h, w = self.mask[index, :]
        label_val = self.gt[h, w]


        if label_val == 0:
            raise ValueError(
                f"Encountered invalid label (0) at position ({h}, {w}). "
                "This suggests a bug in mask generation. "
                "Dataset: {}, Mask length: {}".format(
                    getattr(self, 'dataset_name', 'Unknown'), len(self.mask)
                )
            )

        hsi = self.hsi[h: h + self.patch_size, w: w + self.patch_size]
        lidar = self.lidar[h: h + self.patch_size, w: w + self.patch_size]

        if self.transform:
            hsi = self.transform(hsi).float()
            lidar = self.transform(lidar).float()
            if self.train:
                trans = [transforms.RandomHorizontalFlip(0.5),
                         transforms.RandomVerticalFlip(0.5)]


                i = random.randint(0, 1)
                hsi = trans[i](hsi)
                lidar = trans[i](lidar)


                if self.use_rotation_aug:
                    rot = random.choice([0, 1, 2, 3])
                    if rot > 0:
                        hsi = torch.rot90(hsi, k=rot, dims=[1, 2])
                        lidar = torch.rot90(lidar, k=rot, dims=[1, 2])


        if self.label_map is not None:

            gt_val = self.label_map.get(label_val, 0)
            gt = torch.tensor(gt_val).long()
        else:

            gt = torch.tensor(label_val - 1).long()
        return hsi, lidar, gt

    def __len__(self):
        return self.mask.shape[0]


class PatchDataset(Dataset):


    def __init__(self, hsi, lidar, gt, train=False, label_map=None, use_rotation_aug=False):
        self.hsi = hsi
        self.lidar = lidar
        self.gt = gt
        self.train = train
        self.label_map = label_map
        self.use_rotation_aug = use_rotation_aug

    def __getitem__(self, index):
        hsi = self.hsi[index]
        lidar = self.lidar[index]
        label = self.gt[index]


        hsi = torch.from_numpy(hsi).permute(2, 0, 1).float()


        if lidar.ndim == 2:
            lidar = torch.from_numpy(lidar).unsqueeze(0).float()
        else:
            lidar = torch.from_numpy(lidar).permute(2, 0, 1).float()

        if self.train:

            if random.random() > 0.5:

                hsi = torch.flip(hsi, dims=[2])
                lidar = torch.flip(lidar, dims=[2])
            if random.random() > 0.5:
                hsi = torch.flip(hsi, dims=[1])
                lidar = torch.flip(lidar, dims=[1])


            if self.use_rotation_aug:
                rot = random.choice([0, 1, 2, 3])
                if rot > 0:
                    hsi = torch.rot90(hsi, k=rot, dims=[1, 2])
                    lidar = torch.rot90(lidar, k=rot, dims=[1, 2])


        gt_val = label.item() if hasattr(label, 'item') else label

        if self.label_map is not None:


            gt_val = self.label_map.get(gt_val, 0)
            gt = torch.tensor(gt_val).long()
        else:

            gt = torch.tensor(gt_val - 1).long()

        return hsi, lidar, gt

    def __len__(self):
        return len(self.gt)


class ArrayMinMaxScaler:

    def __init__(self, data_min, data_max):
        self.data_min_ = np.asarray(data_min, dtype=np.float32)
        self.data_max_ = np.asarray(data_max, dtype=np.float32)
        self.scale_ = self.data_max_ - self.data_min_
        self.scale_[self.scale_ == 0] = 1.0
        self.n_features_in_ = int(self.data_min_.shape[0])

    def transform(self, x):
        return (x - self.data_min_) / self.scale_

    def fit_transform(self, x):
        return self.transform(x)


def _houston_scaler_cache_path(mat_path):
    return mat_path + ".minmax_cache.npz"


def _load_houston_scaler_cache(mat_path, hsi_shape, lidar_shape):
    cache_path = _houston_scaler_cache_path(mat_path)
    if not os.path.exists(cache_path):
        return None
    try:
        cache = np.load(cache_path, allow_pickle=False)
        if tuple(cache["hsi_shape"].tolist()) != tuple(hsi_shape):
            return None
        if tuple(cache["lidar_shape"].tolist()) != tuple(lidar_shape):
            return None
        return cache["hsi_min"], cache["hsi_max"], cache["lidar_min"], cache["lidar_max"]
    except Exception as exc:
        print(f"[Houston2018 Lazy] Ignoring invalid scaler cache: {exc}")
        return None


def _save_houston_scaler_cache(mat_path, hsi_shape, lidar_shape, hsi_min, hsi_max, lidar_min, lidar_max):
    cache_path = _houston_scaler_cache_path(mat_path)
    try:
        np.savez(
            cache_path,
            hsi_shape=np.asarray(hsi_shape, dtype=np.int64),
            lidar_shape=np.asarray(lidar_shape, dtype=np.int64),
            hsi_min=hsi_min,
            hsi_max=hsi_max,
            lidar_min=lidar_min,
            lidar_max=lidar_max,
        )
        print(f"[Houston2018 Lazy] Saved scaler cache: {cache_path}")
    except Exception as exc:
        print(f"[Houston2018 Lazy] Failed to save scaler cache: {exc}")


def _infer_houston_layout(shape, has_channel):
    sample_axis = int(np.argmax(shape))
    channel_axis = None
    if has_channel:
        candidates = [i for i in range(len(shape)) if i != sample_axis]
        channel_axis = max(candidates, key=lambda i: shape[i])
    return sample_axis, channel_axis


def _read_houston_block(ds, start, end, sample_axis, channel_axis=None):
    slicer = [slice(None)] * ds.ndim
    slicer[sample_axis] = slice(start, end)
    arr = np.asarray(ds[tuple(slicer)], dtype=np.float32)

    axes = list(range(ds.ndim))
    moved_axes = [sample_axis] + [a for a in axes if a != sample_axis]
    arr = np.moveaxis(arr, sample_axis, 0)

    if channel_axis is not None:
        channel_pos = moved_axes.index(channel_axis)
        arr = np.moveaxis(arr, channel_pos, -1)
    return arr


def _read_houston_sample(ds, sample_idx, sample_axis, channel_axis=None):
    slicer = [slice(None)] * ds.ndim
    slicer[sample_axis] = int(sample_idx)
    arr = np.asarray(ds[tuple(slicer)], dtype=np.float32)

    if channel_axis is not None:
        channel_pos = channel_axis if channel_axis < sample_axis else channel_axis - 1
        arr = np.moveaxis(arr, channel_pos, -1)
    return arr


def _stream_minmax(ds, sample_axis, channel_axis=None, block_size=2048):
    n = ds.shape[sample_axis]
    if channel_axis is None:
        data_min = np.full((1,), np.inf, dtype=np.float32)
        data_max = np.full((1,), -np.inf, dtype=np.float32)
    else:
        data_min = np.full((ds.shape[channel_axis],), np.inf, dtype=np.float32)
        data_max = np.full((ds.shape[channel_axis],), -np.inf, dtype=np.float32)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = _read_houston_block(ds, start, end, sample_axis, channel_axis)
        if channel_axis is None:
            flat = block.reshape(-1, 1)
        else:
            flat = block.reshape(-1, block.shape[-1])
        data_min = np.minimum(data_min, flat.min(axis=0))
        data_max = np.maximum(data_max, flat.max(axis=0))
    return data_min, data_max


class Houston2018LazyPatchDataset(Dataset):


    def __init__(self, mat_path, hsi_key, lidar_key, gt, indices,
                 hsi_sample_axis, hsi_channel_axis, lidar_sample_axis, lidar_channel_axis,
                 hsi_scaler, lidar_scaler, train=False, label_map=None,
                 use_rotation_aug=False, return_indices=False):
        self.mat_path = mat_path
        self.hsi_key = hsi_key
        self.lidar_key = lidar_key
        self.gt = gt
        self.indices = np.asarray(indices, dtype=np.int64)
        self.hsi_sample_axis = hsi_sample_axis
        self.hsi_channel_axis = hsi_channel_axis
        self.lidar_sample_axis = lidar_sample_axis
        self.lidar_channel_axis = lidar_channel_axis
        self.hsi_scaler = hsi_scaler
        self.lidar_scaler = lidar_scaler
        self.train = train
        self.label_map = label_map
        self.use_rotation_aug = use_rotation_aug
        self.return_indices = return_indices
        self._h5 = None

    def _file(self):
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.mat_path, 'r')
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5'] = None
        return state

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

    def __getitem__(self, index):
        sample_idx = int(self.indices[index])
        f = self._file()

        hsi = _read_houston_sample(
            f[self.hsi_key], sample_idx, self.hsi_sample_axis, self.hsi_channel_axis
        )
        lidar = _read_houston_sample(
            f[self.lidar_key], sample_idx, self.lidar_sample_axis, self.lidar_channel_axis
        )

        hsi = self.hsi_scaler.transform(hsi.reshape(-1, hsi.shape[-1])).reshape(hsi.shape)
        if lidar.ndim == 2:
            lidar = self.lidar_scaler.transform(lidar.reshape(-1, 1)).reshape(lidar.shape)
            lidar = torch.from_numpy(lidar).unsqueeze(0).float()
        else:
            lidar = self.lidar_scaler.transform(lidar.reshape(-1, lidar.shape[-1])).reshape(lidar.shape)
            lidar = torch.from_numpy(lidar).permute(2, 0, 1).float()

        hsi = torch.from_numpy(hsi).permute(2, 0, 1).float()

        if self.train:
            if random.random() > 0.5:
                hsi = torch.flip(hsi, dims=[2])
                lidar = torch.flip(lidar, dims=[2])
            if random.random() > 0.5:
                hsi = torch.flip(hsi, dims=[1])
                lidar = torch.flip(lidar, dims=[1])
            if self.use_rotation_aug:
                rot = random.choice([0, 1, 2, 3])
                if rot > 0:
                    hsi = torch.rot90(hsi, k=rot, dims=[1, 2])
                    lidar = torch.rot90(lidar, k=rot, dims=[1, 2])

        label_val = self.gt[sample_idx]
        if self.label_map is not None:
            gt = torch.tensor(self.label_map.get(label_val, 0)).long()
        else:
            gt = torch.tensor(label_val - 1).long()

        if self.return_indices:
            return hsi, lidar, gt, torch.tensor(sample_idx).long()
        return hsi, lidar, gt

    def __len__(self):
        return len(self.indices)


def load_processed_dataset(args):
    hsi_path = os.path.join(args.dataset_dir, args.dataset_name, "HSI.mat")
    hsi = io.loadmat(hsi_path)['HSI'].astype(np.float32)
    H, W, C = hsi.shape
    hsi = hsi.reshape(-1, C)
    scaler_hsi = MinMaxScaler()
    hsi = scaler_hsi.fit_transform(hsi)
    args.scaler_hsi = scaler_hsi

    if args.use_pca:

        pca_model = PCA(n_components=args.pca_component, random_state=42)
        hsi = pca_model.fit_transform(hsi)
        args.pca_model_hsi = pca_model
    hsi = hsi.reshape(H, W, -1)

    if args.dataset_name in ['Augsburg', 'Berlin']:
        lidar_path = os.path.join(args.dataset_dir, args.dataset_name, "SAR.mat")
        lidar = io.loadmat(lidar_path)['SAR'].astype(np.float32)
    else:
        lidar_path = os.path.join(args.dataset_dir, args.dataset_name, "LiDAR.mat")
        lidar = io.loadmat(lidar_path)['LiDAR'].astype(np.float32)

    if lidar.ndim == 2:
        lidar = np.expand_dims(lidar, axis=2)
    H, W, C = lidar.shape
    lidar = lidar.reshape(-1, C)
    scaler_lidar = MinMaxScaler()
    lidar = scaler_lidar.fit_transform(lidar)
    args.scaler_lidar = scaler_lidar
    lidar = lidar.reshape(H, W, -1)

    gt_path = os.path.join(args.dataset_dir, args.dataset_name, "gt.mat")
    gt = io.loadmat(gt_path)['gt'].astype(np.int32)

    return hsi, lidar, gt


def select_mask(args, gt, test_amount=None, train_ratio=None, fold_idx=None, num_folds=None):


    ignored_label = 0
    mask = np.ones(shape=gt.shape, dtype=bool)
    mask[gt == ignored_label] = False
    x_pos, y_pos = np.nonzero(mask)

    m = int(np.max(gt))
    split_seed = getattr(args, 'split_seed', 100)


    default_amounts = {
        'Houston2013': [198, 190, 192, 188, 186, 182, 196, 191, 193, 191, 181, 192, 184, 181, 187],
        'MUUFL': [150] * 11,
        'Trento': [129, 125, 105, 154, 184, 122]
    }
    dataset_amount = default_amounts.get(args.dataset_name, None)

    train_indices_list = []
    val_indices_list = []
    test_indices_list = []

    for i in range(m):

        indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if gt[x, y] == i + 1])
        rng = np.random.RandomState(split_seed)
        rng.shuffle(indices)

        total_samples = len(indices)
        if total_samples == 0: continue


        if train_ratio is not None:
            n_pool = max(2, int(round(total_samples * train_ratio)))
        elif test_amount is not None and i < len(test_amount):
            n_pool = int(test_amount[i])
        elif dataset_amount is not None and i < len(dataset_amount):
            n_pool = int(dataset_amount[i])
        else:
            n_pool = min(200, total_samples // 2)

        pool_idx = indices[:n_pool]
        eval_idx = indices[n_pool:]


        if fold_idx is not None and num_folds is not None:
            f_size = len(pool_idx) // num_folds
            start = fold_idx * f_size
            end = (fold_idx + 1) * f_size if fold_idx != num_folds - 1 else len(pool_idx)

            val_idx = pool_idx[start:end]
            train_idx = np.concatenate([pool_idx[:start], pool_idx[end:]], axis=0)

            train_indices_list.append(train_idx)
            val_indices_list.append(val_idx)
            test_indices_list.append(eval_idx)
        else:

            train_indices_list.append(pool_idx)
            val_indices_list.append(eval_idx)
            test_indices_list.append(eval_idx)

    train_indices = np.vstack(train_indices_list)
    val_indices = np.vstack(val_indices_list)
    test_indices = np.vstack(test_indices_list)

    return None, train_indices, val_indices, test_indices


def make_dataloader(args, test_amount=None, train_ratio=None, fold_idx=None, num_folds=None):


    split_seed = getattr(args, 'split_seed', getattr(args, 'seed', 100))
    np.random.seed(split_seed)
    random.seed(split_seed)

    if args.dataset_name == 'Houston2018':
        data_dir = os.path.join(args.dataset_dir, args.dataset_name)
        target_file = "Houston2018.mat"
        m_path = os.path.join(data_dir, target_file)

        if args.use_pca:
            raise ValueError("Houston2018 lazy loader keeps RAM low and does not support PCA materialization. Run with --use_pca False.")

        import h5py

        def _find_key(keys, *needles):
            matches = [k for k in keys if any(n in k.lower() for n in needles)]
            if not matches:
                raise KeyError(f"Cannot find key containing {needles} in {list(keys)}")
            return matches[0]

        with h5py.File(m_path, 'r') as m:
            keys = list(m.keys())
            h_key = _find_key(keys, 'subhsi', 'hsi')
            l_key = _find_key(keys, 'sublidar', 'lidar')
            g_key = _find_key(keys, 'act_y', 'gt', 'label', 'y')
            i_key = _find_key(keys, 'indexi')
            j_key = _find_key(keys, 'indexj')

            gt = np.asarray(m[g_key]).flatten().astype(np.int64)
            idx_i = np.asarray(m[i_key]).flatten().astype(np.int64)
            idx_j = np.asarray(m[j_key]).flatten().astype(np.int64)

            hsi_sample_axis, hsi_channel_axis = _infer_houston_layout(m[h_key].shape, has_channel=True)
            lidar_has_channel = (m[l_key].ndim == 4)
            lidar_sample_axis, lidar_channel_axis = _infer_houston_layout(m[l_key].shape, has_channel=lidar_has_channel)

            sample_hsi = _read_houston_sample(m[h_key], 0, hsi_sample_axis, hsi_channel_axis)
            if sample_hsi.ndim != 3:
                raise ValueError(f"Unexpected Houston2018 HSI sample shape: {sample_hsi.shape}")
            pH, pW, hsi_channels = sample_hsi.shape
            if pH != pW:
                print(f"[Warning] Non-square Houston2018 patches detected: {pH}x{pW}")
            if args.patch_size != pH:
                print(f"[Houston2018] args.patch_size ({args.patch_size}) -> data patch size ({pH})")
                args.patch_size = pH

            print(f"[Houston2018 Lazy] HSI key={h_key}, shape={m[h_key].shape}, sample_axis={hsi_sample_axis}, channel_axis={hsi_channel_axis}")
            print(f"[Houston2018 Lazy] LiDAR key={l_key}, shape={m[l_key].shape}, sample_axis={lidar_sample_axis}, channel_axis={lidar_channel_axis}")

            cache = _load_houston_scaler_cache(m_path, m[h_key].shape, m[l_key].shape)
            if cache is None:
                print("[Houston2018 Lazy] Streaming min/max scalers without materializing full arrays...")
                hsi_min, hsi_max = _stream_minmax(m[h_key], hsi_sample_axis, hsi_channel_axis)
                lidar_min, lidar_max = _stream_minmax(m[l_key], lidar_sample_axis, lidar_channel_axis)
                _save_houston_scaler_cache(m_path, m[h_key].shape, m[l_key].shape, hsi_min, hsi_max, lidar_min, lidar_max)
            else:
                print(f"[Houston2018 Lazy] Loaded scaler cache: {_houston_scaler_cache_path(m_path)}")
                hsi_min, hsi_max, lidar_min, lidar_max = cache

        args.scaler_hsi = ArrayMinMaxScaler(hsi_min, hsi_max)
        args.scaler_lidar = ArrayMinMaxScaler(lidar_min, lidar_max)
        args.houston2018_lazy = True
        args.houston2018_mat_path = m_path
        args.houston2018_hsi_key = h_key
        args.houston2018_lidar_key = l_key
        args.houston2018_gt_key = g_key
        args.houston2018_indexi_key = i_key
        args.houston2018_indexj_key = j_key
        args.houston2018_hsi_sample_axis = hsi_sample_axis
        args.houston2018_hsi_channel_axis = hsi_channel_axis
        args.houston2018_lidar_sample_axis = lidar_sample_axis
        args.houston2018_lidar_channel_axis = lidar_channel_axis
        args.houston2018_hsi_channels = int(hsi_channels)

        total_samples = len(gt)
        labels_in_gt, counts = np.unique(gt, return_counts=True)
        threshold = 0.002 * total_samples
        unique_labels = sorted([int(l) for l, c in zip(labels_in_gt, counts) if l > 0 and c >= threshold])

        args.num_classes = len(unique_labels)
        args.original_labels = unique_labels
        args.label_map = {old: new for new, old in enumerate(unique_labels)}
        args.reverse_label_map = {new: old for new, old in enumerate(unique_labels)}

        all_indices = np.arange(len(gt))
        train_indices, val_indices, test_indices = [], [], []

        for label in unique_labels:
            l_idx = all_indices[gt == label]
            rng = np.random.RandomState(split_seed)
            rng.shuffle(l_idx)

            n_pool = max(2, int(round(len(l_idx) * train_ratio))) if train_ratio else 20
            pool = l_idx[:n_pool]
            eval_idx = l_idx[n_pool:]

            if fold_idx is not None and num_folds is not None:
                f_size = len(pool) // num_folds
                start = fold_idx * f_size
                end = (fold_idx + 1) * f_size if fold_idx != num_folds - 1 else len(pool)
                v_idx = pool[start:end]
                t_idx = np.concatenate([pool[:start], pool[end:]], axis=0)
                train_indices.extend(t_idx)
                val_indices.extend(v_idx)
                test_indices.extend(eval_idx)
            else:
                train_indices.extend(pool)
                val_indices.extend(eval_idx)
                test_indices.extend(eval_idx)

        train_datasets = Houston2018LazyPatchDataset(
            m_path, h_key, l_key, gt, train_indices,
            hsi_sample_axis, hsi_channel_axis, lidar_sample_axis, lidar_channel_axis,
            args.scaler_hsi, args.scaler_lidar, train=True,
            label_map=args.label_map, use_rotation_aug=getattr(args, 'use_rotation_aug', False)
        )
        val_datasets = Houston2018LazyPatchDataset(
            m_path, h_key, l_key, gt, val_indices,
            hsi_sample_axis, hsi_channel_axis, lidar_sample_axis, lidar_channel_axis,
            args.scaler_hsi, args.scaler_lidar, train=False, label_map=args.label_map
        )
        test_datasets = Houston2018LazyPatchDataset(
            m_path, h_key, l_key, gt, test_indices,
            hsi_sample_axis, hsi_channel_axis, lidar_sample_axis, lidar_channel_axis,
            args.scaler_hsi, args.scaler_lidar, train=False, label_map=args.label_map
        )

    else:
        hsi, lidar, gt = load_processed_dataset(args)


        data_dir = os.path.join(args.dataset_dir, args.dataset_name)
        tr_path = os.path.join(data_dir, "TRLabel.mat")
        ts_path = os.path.join(data_dir, "TSLabel.mat")

        if os.path.exists(tr_path) and os.path.exists(ts_path) and train_ratio is None:
            print(f"Detected predefined split for {args.dataset_name}. Loading TRLabel and TSLabel...")
            tr_gt = io.loadmat(tr_path)['TRLabel'].astype(np.int32)
            ts_gt = io.loadmat(ts_path)['TSLabel'].astype(np.int32)


            train_indices = np.array(np.nonzero(tr_gt)).T
            test_indices = np.array(np.nonzero(ts_gt)).T


            val_indices = test_indices


            unique_labels = np.unique(gt[gt > 0])
            args.num_classes = len(unique_labels)
            args.label_map = {old: new for new, old in enumerate(unique_labels)}
            args.reverse_label_map = {new: old for new, old in enumerate(unique_labels)}
            print(f"Predefined split: {args.num_classes} classes mapping established.")
        else:

            total_pixels = gt.size
            unique_labels_in_gt, counts = np.unique(gt, return_counts=True)
            threshold = 0.002 * total_pixels
            unique_labels = []
            for l, c in zip(unique_labels_in_gt, counts):
                if l > 0 and c >= threshold: unique_labels.append(l)

            unique_labels = sorted(unique_labels)
            args.num_classes = len(unique_labels)
            args.original_labels = unique_labels
            args.label_map = {old: new for new, old in enumerate(unique_labels)}
            args.reverse_label_map = {new: old for new, old in enumerate(unique_labels)}


            _, train_indices, val_indices, test_indices = select_mask(
                args, gt, test_amount=test_amount, train_ratio=train_ratio,
                fold_idx=fold_idx, num_folds=num_folds
            )

        train_datasets = HXDataset(args, hsi, lidar, gt, train_indices, train=True, label_map=args.label_map)
        val_datasets = HXDataset(args, hsi, lidar, gt, val_indices, train=False, label_map=args.label_map)
        test_datasets = HXDataset(args, hsi, lidar, gt, test_indices, train=False, label_map=args.label_map)


    sampler = None
    if getattr(args, 'balanced_sampling', False):
        from collections import Counter
        if getattr(args, 'houston2018_lazy', False):
            lbls = [args.label_map.get(int(train_datasets.gt[int(i)]), 0) for i in train_datasets.indices]
        else:
            lbls = []
            for i in range(len(train_datasets)):
                _, _, gt = train_datasets[i]
                lbls.append(int(gt.item()))
        counts = Counter(lbls)
        print(f"[BalancedSampling] Train label counts: {counts}")
        weights = [1.0 / float(counts[int(l)]) for l in lbls]
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        print("[BalancedSampling] Enabled: using WeightedRandomSampler to generate balanced batches.")

    num_workers = 0 if getattr(args, 'houston2018_lazy', False) else 4
    if sampler is not None:
        train_loader = DataLoader(train_datasets, batch_size=args.batch_size, num_workers=num_workers, sampler=sampler)
    else:
        train_loader = DataLoader(train_datasets, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)


    val_bs = getattr(args, 'val_batch_size', 2048)

    val_loader = DataLoader(val_datasets, batch_size=val_bs, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_datasets, batch_size=val_bs, num_workers=num_workers, shuffle=False)

    print(f"Data Split -> Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)} | Val BS: {val_bs} Test BS: {val_bs}")
    return train_loader, val_loader, test_loader
