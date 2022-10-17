from typing import Any, Callable

from pl_bolts.datamodules import ImagenetDataModule, CIFAR10DataModule
from pl_bolts.datasets import UnlabeledImagenet
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, Flowers102

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class FewShotImagenetDataModule(ImagenetDataModule):
    name = "few-shot-imagenet"

    def __init__(self, label_pct, **kwargs):
        """
        Args:
            label_pct: % of labels for training
        """
        super().__init__(**kwargs)
        self.label_pct = label_pct
        self.num_samples = int(label_pct / 100 * self.num_samples)

        # just to silence the error from the linter
        self.train_transforms = None

    def train_dataloader(self) -> DataLoader:
        """
        Uses the train split of imagenet2012, puts away a portion of it
        for the validation split, and samples `top-n`% of labeled
        training set in class-balanced way."""
        if self.train_transforms is None:
            transforms = self.train_transform()
        else:
            transforms = self.train_transforms

        dataset = UnlabeledImagenet(
            self.data_dir,
            num_imgs_per_class=self.num_samples // self.num_classes,
            num_imgs_per_class_val_split=self.num_imgs_per_val_class,
            meta_dir=self.meta_dir,
            split="train",
            transform=transforms,
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader


class CIFAR100DataModule(CIFAR10DataModule):
    name = "cifar100"
    dataset_cls = CIFAR100

    @property
    def num_classes(self) -> int:
        return 100


class Flowers102DataModule(LightningDataModule):
    """
    Oxford 102 Flowers train, val and test dataloaders.
    """

    name = "flowers102"

    def __init__(
            self,
            data_dir: str,
            image_size: int = 224,
            num_workers: int = 0,
            batch_size: int = 32,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: where to save/load the data
            image_size: final image size
            num_workers: how many workers to use for loading data
            batch_size: the batch size
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use Flower102 dataset loaded from `torchvision` "
                "which is not installed yet."
            )

        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.train_transforms = None
        self.val_transforms = None

    @property
    def num_classes(self) -> int:
        return 102

    def prepare_data(self) -> None:
        Flowers102(self.data_dir, download=True)

    @staticmethod
    def normalization() -> Callable:
        return transform_lib.Normalize(
            mean=[x / 255.0 for x in [110.4, 97.4, 75.6]],
            std=[x / 255.0 for x in [15.3, 14.8, 20.5]],
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_transforms is None:
            transforms = self.train_transform()
        else:
            transforms = self.train_transforms

        dataset = Flowers102(self.data_dir, split='train', transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader

    def val_dataloader(self) -> DataLoader:
        if self.val_transforms is None:
            transforms = self.val_transform()
        else:
            transforms = self.val_transforms

        dataset = Flowers102(self.data_dir, split='val', transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader

    def test_dataloader(self) -> DataLoader:
        if self.val_transforms is None:
            transforms = self.val_transform()
        else:
            transforms = self.val_transforms

        dataset = Flowers102(self.data_dir, split='test', transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader

    def train_transform(self) -> Callable:
        preprocessing = transform_lib.Compose(
            [
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                self.normalization(),
            ]
        )

        return preprocessing

    def val_transform(self) -> Callable:
        preprocessing = transform_lib.Compose(
            [
                transform_lib.Resize(int(self.image_size + 0.1 * self.image_size)),
                transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                self.normalization(),
            ]
        )

        return preprocessing
