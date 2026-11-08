import os
import json
import torch
import numpy as np
from typing import List, Dict, Union, Callable, Any
from torchvision.transforms import (RandomApply, Compose, RandomChoice,
                                    RandomGrayscale, RandomResizedCrop,
                                    ColorJitter, RandomHorizontalFlip,
                                    ToTensor, Normalize)
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from collections.abc import Iterable
# Local imports
from utils import GaussianBlur
from dataset.affect.get_data import Affect, collate_fn_timeseries
from dataset.mimic.get_data import MIMIC
from dataset.robotics.multimodal_manipulation import (
    MultimodalManipulationDataset, ProcessForce, ProcessImage)


def _normalize_labeled_ratio(labeled_ratio: Union[int, float, str]) -> float:
    ratio = float(labeled_ratio)
    if ratio > 1.0:
        ratio = ratio / 100.0
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("labeled_ratio must be in [0, 1] or a percentage in [0, 100].")
    return ratio


def _label_key(label: Any):
    label = np.asarray(label).reshape(-1)
    if label.size == 1:
        return label.item()
    return tuple(label.tolist())


def _build_labeled_mask(labels: List[Any], labeled_ratio: float, label_seed: int) -> torch.Tensor:
    n_samples = len(labels)
    if labeled_ratio >= 1.0:
        return torch.ones(n_samples, dtype=torch.bool)
    if labeled_ratio <= 0.0:
        return torch.zeros(n_samples, dtype=torch.bool)

    rng = np.random.default_rng(int(label_seed))
    mask = np.zeros(n_samples, dtype=bool)
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(_label_key(label), []).append(idx)

    # Use a stratified split for classification-style labels; fall back to a
    # plain random split when labels are mostly unique, as in regression tasks.
    use_stratified = 1 < len(label_to_indices) <= max(2, n_samples // 2)
    if use_stratified:
        for indices in label_to_indices.values():
            indices = np.asarray(indices)
            n_labeled = int(round(len(indices) * labeled_ratio))
            n_labeled = min(len(indices), max(1, n_labeled))
            selected = rng.permutation(indices)[:n_labeled]
            mask[selected] = True
    else:
        n_labeled = int(round(n_samples * labeled_ratio))
        n_labeled = min(n_samples, max(1, n_labeled))
        selected = rng.permutation(n_samples)[:n_labeled]
        mask[selected] = True

    return torch.from_numpy(mask)


class MultiBenchDataModule(LightningDataModule):
    """Create MultiBench data loaders, supporting currently 6 datasets:
        - MIMIC (tabular + time-series) with n=36k [health recordings + patient backgrounds info]
            Shape: tabular=(*, 5), time-series=(*, 24, 12), label=(*,) (binary 0 or 1)
        - MOSEI (vision, text, audio) with n=22k [sentiment prediction from videos]
            Shape: vision=(*, T, 35), audio=(*, T, 74), text=(*, T, 300), label=(*,) (continuous between [-3, 3])
        - MOSI (vision, text, audio) with n=2k [sentiment prediction from videos]
            Shape: vision=(*, T, 20), audio=(*, T, 74), text=(*, T, 300), label=(*,) (continuous between [-3, 3])
        - UR-FUNNY (vision, text, audio) with n=16k [humor prediction from videos]
            Shape: vision=(*, T, 371), audio=(*, T, 81), text=(*, T, 300), label=(*,) (binary 0 or 1)
        - MUSTARD (vision, text, audio) with n=690 [sarcasm prediction from videos]
            Shape: vision=(*, T, 371), audio=(*, T, 81), text=(*, T, 300), label=(*,) (binary 0 or 1)
        - VISION&TOUCH (vision, force, proprioception) with n=117k (train) + 29k (test)
            Shape: vision=(*, 3, 128, 128), force=(*, T, 6) [after truncation], proprio=(*, 8),
                label=(*, 4) if task=="ee_yaw_next" (continuous) or (*,) if task=="contact_next" (binary)

        Sequence T varies for each sample in each dataset, up to 50 (max padding length).
        Each batch of data consists of pairs (X, y) where X is a tuple/list of `n` modalities and y is a Tensor.
        Each modality is a Tensor of shape (*, T, p) or (*, p) depending on the modality (sequence or tabular data)
    """
    def __init__(self, dataset: str,
                 model: str,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 labeled_ratio: float = 1.0,
                 label_seed: int = 0,
                 **kwargs):
        """
        Args:
            dataset: in {"mimic", "mosi", "mosei", "humor", "sarcasm", "visionandtouch", "visionandtouch-bin"}
            model: in {'Sup', 'SupervisedClassifier','MIIB', 'CLIP', 'CMC', 'CrossSelf'}
                The model defines the augmentations to apply:
                    - Sup, SupervisedClassifier: no augmentation, returns the modalities + label
                    - CLIP, CMC: no augmentation, returns the modalities without labels
                    - MIIB: augmentation for each modality, returns pairs of augmented modalities
                    - CrossSelf: augmentation + original modality
            batch_size: Batch size given to dataloader (train, val, test)
            num_workers: Number of CPU workers for data loading
            labeled_ratio: Fraction of labeled samples in the training split.
                Values in [0, 1] are ratios; values >1 are interpreted as percentages.
                Only SupervisedClassifier fine-tuning uses this.
            label_seed: Seed used to choose the labeled subset.
            kwargs: keyword args given to the chosen torch `Dataset`
        """
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.labeled_ratio = _normalize_labeled_ratio(labeled_ratio)
        self.label_seed = int(label_seed)
        self.dataset_kwargs = kwargs

        if self.model == "Sup":
            self.train_dataset = MultiBench(self.dataset, "train", **self.dataset_kwargs)
            self.val_dataset = MultiBench(self.dataset, "val", **self.dataset_kwargs)
            self.test_dataset = MultiBench(self.dataset, "test", **self.dataset_kwargs)
        elif self.model == "CLIP" or self.model == "CMC":
            self.train_dataset = MultiBenchCLIP(self.dataset, "train", **self.dataset_kwargs)
            self.val_dataset = MultiBenchCLIP(self.dataset, "val", **self.dataset_kwargs)
            self.test_dataset = MultiBenchCLIP(self.dataset, "test", **self.dataset_kwargs)
        elif self.model == "MIIB":
            self.train_dataset = MultiBenchSSL(self.dataset, "train", **self.dataset_kwargs)
            self.val_dataset = MultiBenchSSL(self.dataset, "val", **self.dataset_kwargs)
            self.test_dataset = MultiBenchSSL(self.dataset, "test", **self.dataset_kwargs)
        elif self.model == "CrossSelf":
            self.train_dataset = MultiBenchCrossSelf(self.dataset, "train", **self.dataset_kwargs)
            self.val_dataset = MultiBenchCrossSelf(self.dataset, "val", **self.dataset_kwargs)
            self.test_dataset = MultiBenchCrossSelf(self.dataset, "test", **self.dataset_kwargs)
        elif self.model == "SupervisedClassifier":
            if dataset == "visionandtouch":
                self.train_dataset = MultiBench(self.dataset, "train", **self.dataset_kwargs)
                self.val_dataset = MultiBench(self.dataset, "val", **self.dataset_kwargs)
                self.test_dataset = MultiBench(self.dataset, "test", **self.dataset_kwargs)
            else:
                # self.train_dataset = MultiBench(
                #     self.dataset, "train",
                #     labeled_ratio=self.labeled_ratio,
                #     label_seed=self.label_seed,
                #     return_label_mask=self.labeled_ratio < 1.0,
                #     **self.dataset_kwargs)
                self.train_dataset = MultiBenchSupCon(self.dataset, "train", **self.dataset_kwargs)
                self.val_dataset = MultiBenchSupCon(self.dataset, "val", **self.dataset_kwargs)
                self.test_dataset = MultiBenchSupCon(self.dataset, "test", **self.dataset_kwargs)
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, collate_fn=self.train_dataset.collate_fn,
            shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=self.val_dataset.collate_fn,
            shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, collate_fn=self.test_dataset.collate_fn,
            shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False)


class MultiBench(Dataset):
    def __init__(self, dataset: str, split: str, **dataset_kwargs):
        super().__init__()
        from dataset.catalog import load_catalog
        self.catalog = load_catalog()
        self.collate_fn = None
        if dataset == "mimic":
            self.dataset = MIMIC(self.catalog[dataset]["path"], split=split, **dataset_kwargs)
        elif dataset in ["humor", "sarcasm", "mosi", "mosei"]:
            self.dataset = Affect(self.catalog[dataset]["path"],
                                  dataset=dataset,
                                  split=split, **dataset_kwargs)
            self.collate_fn = self.collate_fn_affect
        elif dataset in ["visionandtouch", "visionandtouch-bin"]:
            dataset = "visionandtouch"
            self.dataset = MultimodalManipulationDataset(
                self.catalog[dataset]["path"],
                split=split,
                transform=Compose(
                    [
                        ProcessForce(32, "force", tanh=True),
                        ProcessImage(128)
                    ]),
                **dataset_kwargs)
        else:
            raise ValueError(f"Dataset not implemented: {dataset}")
        self.modalities = self.dataset.modalities

    @staticmethod
    def collate_fn_affect(inputs: List, max_seq_length: int = None):
        """Handles a list of labelled timeseries data with eventually different lengths.
        Args:
             inputs: list (X, y) where X is a list of modalities and y a label
             max_seq_length: if set, pads all timeseries to `max_seq_length`.
                Otherwise, all sequences are padded to the maximum sequence length in this batch.
        Output:
            (X, y) where X is a list of modalities with shape (*, T, p) and y a label with shape (*, 1).
            If `max_seq_length` is set then T == `max_seq_length`.
        """
        labels = np.array([y for (X, y) in inputs]).reshape(len(inputs),)
        X = collate_fn_timeseries([X for (X, y) in inputs], max_seq_length)
        return X, torch.tensor(labels)

    def __getitem__(self, i):
        X, y = self.dataset[i]
        return X, y

    def __len__(self):
        return len(self.dataset)


class MultiBenchCLIP(MultiBench):
    """Returns a list of modalities without labels."""
    def __getitem__(self, i):
        X, y = self.dataset[i]
        return X

    @staticmethod
    def collate_fn_affect(inputs: List, max_seq_length: int = None):
        return collate_fn_timeseries(inputs, max_seq_length)


class MultiBenchSupCon(MultiBench):
    """Apply data augmentation to all modalities to create pairs X1, X2 from a list of modalities X."""
    def __init__(self, dataset: str, split: str,
                 labeled_ratio: float = 1.0,
                 label_seed: int = 0,
                 return_label_mask: bool = False,
                 **dataset_kwargs):
        aug_kwargs = MultiBenchAugmentations.parse_kwargs(dataset_kwargs)
        super().__init__(dataset, split, **dataset_kwargs)
        self.labeled_ratio = _normalize_labeled_ratio(labeled_ratio)
        self.return_label_mask = return_label_mask and self.labeled_ratio < 1.0
        self.label_mask = None
        if self.return_label_mask:
            labels = [self.dataset[i][1] for i in range(len(self.dataset))]
            self.label_mask = _build_labeled_mask(labels, self.labeled_ratio, label_seed)
        self.augment = MultiBenchAugmentations(**aug_kwargs)
        if isinstance(self.dataset, MultimodalManipulationDataset): # removes img transform
            self.dataset.transform.transforms = self.dataset.transform.transforms[:-1]

    def __getitem__(self, i):
        X, y = self.dataset[i]
        aug1 = self.augment(X) # augment all modalities together
        aug2 = self.augment(X)
        if self.return_label_mask:
            return aug1, aug2, y, bool(self.label_mask[i])
        return aug1, aug2, y

    @staticmethod
    def collate_fn_affect(inputs: List, max_seq_length: int = None):
        if len(inputs[0]) == 4:
            aug1 = collate_fn_timeseries([X1 for (X1, X2, y, mask) in inputs], max_seq_length)
            aug2 = collate_fn_timeseries([X2 for (X1, X2, y, mask) in inputs], max_seq_length)
            Y = torch.tensor([y for (X1, X2, y, mask) in inputs])
            label_mask = torch.tensor([mask for (X1, X2, y, mask) in inputs], dtype=torch.bool)
            return aug1, aug2, Y, label_mask

        aug1 = collate_fn_timeseries([X1 for (X1, X2, y) in inputs], max_seq_length)
        aug2 = collate_fn_timeseries([X2 for (X1, X2, y) in inputs], max_seq_length)
        Y = torch.tensor([y for (X1, X2, y) in inputs])
        return aug1, aug2, Y


class MultiBenchSSL(MultiBench):
    """Apply data augmentation to all modalities to create pairs X1, X2 from a list of modalities X."""
    def __init__(self, dataset: str, split: str, **dataset_kwargs):
        aug_kwargs = MultiBenchAugmentations.parse_kwargs(dataset_kwargs)
        super().__init__(dataset, split, **dataset_kwargs)
        self.augment = MultiBenchAugmentations(**aug_kwargs)
        if isinstance(self.dataset, MultimodalManipulationDataset): # removes img transform
            self.dataset.transform.transforms = self.dataset.transform.transforms[:-1]

    def __getitem__(self, i):
        X, y = self.dataset[i]
        aug1 = self.augment(X) # augment all modalities together
        aug2 = self.augment(X)
        return aug1, aug2

    @staticmethod
    def collate_fn_affect(inputs: List, max_seq_length: int = None):
        aug1 = collate_fn_timeseries([X1 for (X1, X2) in inputs], max_seq_length)
        aug2 = collate_fn_timeseries([X2 for (X1, X2) in inputs], max_seq_length)
        return aug1, aug2


class MultiBenchCrossSelf(MultiBench):
    """Apply data augmentation to each modality to create augmented pairs X1, X2 + return original input X"""
    def __init__(self, dataset: str, split: str, **dataset_kwargs):
        aug_kwargs = MultiBenchAugmentations.parse_kwargs(dataset_kwargs)
        super().__init__(dataset, split, **dataset_kwargs)
        self.augment = MultiBenchAugmentations(**aug_kwargs)
        self.img_transform = None
        if isinstance(self.dataset, MultimodalManipulationDataset):  # removes img transform
            self.img_transform = self.dataset.transform.transforms.pop(-1)

    def __getitem__(self, i):
        X, _ = self.dataset[i]
        assert len(X) == 2, f"Not implemented for {len(X)} modalities"
        aug1 = self.augment(X)
        aug2 = self.augment(X)
        if self.img_transform is not None: # small hack
            self.dataset.transform.transforms.append(self.img_transform)
            X, _ = self.dataset[i]
            self.img_transform = self.dataset.transform.transforms.pop(-1)
        return X[0], X[1], [aug1[0], aug2[0]], [aug1[1], aug2[1]]

    @staticmethod
    def collate_fn_affect(inputs: List, max_seq_length: int = None):
        X = collate_fn_timeseries([X for (X, X1, X2) in inputs], max_seq_length)
        aug1 = collate_fn_timeseries([X1 for (X, X1, X2) in inputs], max_seq_length)
        aug2 = collate_fn_timeseries([X2 for (X, X1, X2) in inputs], max_seq_length)
        return X, aug1, aug2


class AugMapper:
    """ Map a list of modalities X to a list of
        augmented modalities X' through a list of transformations T
        such that X' = [T[0](X[0)), ..., T[n](X[n])]
    """

    def __init__(self, tfs: List[Callable]):
        self.transforms = tfs

    def __call__(self, x: List[Any]):
        assert len(x) == len(self.transforms), "Number of modalities must match number of transformations"
        return [tf([x[i]])[0] for i, tf in enumerate(self.transforms)]


class SimCLRAug:
    """Apply SimCLR augmentation to an input image."""
    def __init__(self, size=224):
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        self.aug = Compose([
            RandomResizedCrop(size, scale=(0.08, 1.)),
            RandomApply([
                ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            RandomGrayscale(p=0.2),
            RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])

    def __call__(self, list_x):
        x_tf = []
        for x in list_x:
            x_tf.append(self.aug(x))
        return x_tf


class MultiBenchAugmentations(torch.nn.Module):
    """ Defines a set of augmentations to apply to time-series/tabular data based on [1] and
        to images based on [2]
    [1] Factorized Contrastive Learning: Going Beyond Multi-view Redundancy, Liang et al., NeurIPS 2023
    [2] A Simple Framework for Contrastive Learning of Visual Representations, Chen et al., ICML 2020
    """

    def __init__(self, augmentations: Union[str, List[str]] = None,
                 p: float = 0.5,
                 random_choice: bool = False):
        """
        :param p: probability for applying each augmentation individually if `random_choice` is false
        :param augmentations: transformations to apply either:
            - individually to each modality (if List) => unimodal aug
            - together on all modalities (if str) => multi-modal aug
        :param random_choice: randomly choose one transformation to apply
        """
        super().__init__()
        if augmentations is None:
            augmentations = []
        assert isinstance(augmentations, str) or isinstance(augmentations, Iterable), \
            f"Unknown type: {type(augmentations)}"
        if isinstance(augmentations, str):
            augmentations = [augmentations]
        transforms = []
        for augmentations_ in augmentations:
            augmentations_ = augmentations_.split("+")
            tf = []
            for aug in augmentations_:
                if aug == "permute":
                    tf.append(self.permute)
                elif aug == "noise":
                    tf.append(self.noise)
                elif aug in ["drop", "multi_drop"]:
                    tf.append(lambda x: self.drop(x, multimodal=(aug=="multi_drop")))
                elif aug in ["drop_consecutive", "multi_drop_consecutive"]:
                    tf.append(lambda x: self.drop_consecutive(x, multimodal=(aug=="multi_drop_consecutive")))
                elif aug in ["crop", "multi_crop"]:
                    tf.append(lambda x: self.crop(x, multimodal=(aug=="multi_crop")))
                elif aug == "mixup":
                    tf.append(self.mixup)
                elif aug == "simclr":
                    tf.append(SimCLRAug(size=128))
                else:
                    raise ValueError(f"Unknown augmentation: {aug}")

            if random_choice:
                transforms.append(RandomChoice(tf))
            else:
                transforms.append(Compose([RandomApply([tf_], p=p)
                                           if not isinstance(tf_, SimCLRAug) else tf_ for tf_ in tf]))

        if len(transforms) == 1:
            self.transforms = transforms[0]
        elif len(transforms) > 1:
            self.transforms = AugMapper(transforms) # map each tf to a modality
        else:
            self.transforms = lambda x: x

    @staticmethod
    def parse_kwargs(kwargs: Dict):
        """Parse and return keywords arguments relevant for this class.
        It is remove in-place from input `kwargs`"""
        aug_kwargs = {}
        if "augmentations" in kwargs:
            aug_kwargs["augmentations"] = kwargs.pop("augmentations")
        if "random_choice" in kwargs:
            aug_kwargs["random_choice"] = kwargs.pop("random_choice")
        return aug_kwargs

    def forward(self, x):
        # x: List[np.ndarray]
        #  List of arrays (one per modality) of shape (T, p)
        #  where `T`==seq length and `p`==num features
        return self.transforms(x)

    @staticmethod
    def permute(x, multimodal=True):
        if len(x) > 0:
            # shuffle the sequence order
            if multimodal:
                idx = np.random.permutation(x[0].shape[0])
                return [x_[idx] for x_ in x]
            x = [x_[np.random.permutation(x_.shape[0])] for x_ in x]
        return x

    @staticmethod
    def noise(x, std=0.1):
        return [x_ + np.random.randn(*x_.shape).astype(np.float32) * std for x_ in x]

    @staticmethod
    def drop(x, frac=(0, 0.8), multimodal=True):
        # drop from 0% to 80% of the sequences
        def get_drop(x_):
            frac_ = np.random.uniform(*frac)
            drop_num = round(frac_ * len(x_))
            drop_idxs = np.random.choice(len(x_), drop_num, replace=False)
            return drop_idxs
        if len(x) > 0:
            if multimodal:
                drop_idxs = get_drop(x[0])
            x_aug = []
            for x_ in x:
                x_aug_ = np.copy(x_)
                if not multimodal:
                    drop_idxs = get_drop(x_)
                x_aug_[drop_idxs] = 0.0
                x_aug.append(x_aug_)
            return x_aug
        return x

    @staticmethod
    def drop_consecutive(x, frac=(0, 0.8), multimodal=True):
        def get_drop(x_):
            frac_ = np.random.uniform(*frac)
            drop_num = round(frac_ * len(x_))
            start_idx = np.random.randint(0, max(len(x_) - drop_num, 1))
            return start_idx, drop_num
        # drop consecutively from 0% to 80% of the sequence
        if len(x) > 0:
            if multimodal:
                start_idx, drop_num = get_drop(x[0])
            x_aug = []
            for x_ in x:
                x_aug_ = np.copy(x_)
                if not multimodal:
                    start_idx, drop_num = get_drop(x_)
                x_aug_[start_idx:start_idx+drop_num] = 0.0
                x_aug.append(x_aug_)
            return x_aug
        return x

    @staticmethod
    def crop(x, size=(0.08, 1), multimodal=True):
        # crop from 8% to 100% of the sequence
        def get_crop(x_):
            size_ = np.random.uniform(*size)
            crop_num = round(size_ * len(x_))
            start_idx = np.random.randint(0, max(len(x_) - crop_num, 1))
            return start_idx, crop_num
        if len(x) > 0:
            if multimodal:
                start_idx, crop_num = get_crop(x[0])
            x_aug = []
            for x_ in x:
                x_aug_ = np.copy(x_)
                if not multimodal:
                    start_idx, crop_num = get_crop(x_)
                x_aug_[:start_idx] = 0.0
                x_aug_[start_idx + crop_num:] = 0.0
                x_aug.append(x_aug_)
            return x_aug
        return x

    @staticmethod
    def mixup(x, alpha=1.0):
        if len(x) > 0:
            indices = np.random.permutation(x[0].shape[0])
            lam = np.random.beta(alpha, alpha)
            x = [x_ * lam + x_[indices] * (1 - lam) for x_ in x]
        return x
