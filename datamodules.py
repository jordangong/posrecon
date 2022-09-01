from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.datasets import UnlabeledImagenet
from torch.utils.data import DataLoader


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
