from argparse import ArgumentParser, BooleanOptionalAction

import torch
import torch.nn.functional as F
from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.models.self_supervised import SSLFineTuner
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRFinetuneTransform
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from datamodules import FewShotImagenetDataModule
from main import PosReconCLR


class PosReconCLREval(SSLFineTuner):
    def __init__(
        self,
        protocol: str = 'linear',
        label_smoothing: float = 0.,
        optim: str = 'sgd',
        **kwargs
    ):
        """
        Args:
            protocol: evaluation protocol, including `linear` and `finetune`
            label_smoothing: label smoothing regularization
            optim: optimizer (SGD or Adam)
        """
        assert protocol in {'linear', 'finetune'}, f"unknown protocol: {protocol}"
        assert optim in {'sgd', 'adam'}, f"unknown optimizer: {optim}"

        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=['backbone'])

        self.protocol = protocol
        self.label_smoothing = label_smoothing
        self.optim = optim

    def on_train_epoch_start(self) -> None:
        if self.protocol == 'linear':
            self.backbone.eval()
        elif self.protocol == 'finetune':
            self.backbone.train()

    def forward_backbone(self, x):
        x = self.backbone.patch_embed(x)
        x += self.backbone.pos_embed

        cls_tokens = self.backbone.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        return x[:, 0, :]

    def shared_step(self, batch):
        x, y = batch

        if self.protocol == 'linear':
            with torch.no_grad():
                feats = self.forward_backbone(x)
        elif self.protocol == 'finetune':
            feats = self.forward_backbone(x)

        logits = self.linear_layer(feats)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.train_acc(logits.softmax(-1), y)

        self.log(f"loss/xent_{self.protocol}/train", loss, prog_bar=True)
        self.log(f"acc/{self.protocol}/train_step", acc, prog_bar=True)
        self.log(f"acc/{self.protocol}/train_epoch", self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.val_acc(logits.softmax(-1), y)

        self.log(f"loss/xent_{self.protocol}/val",
                 loss, prog_bar=True, sync_dist=True)
        self.log(f"acc/{self.protocol}/val_epoch", self.val_acc)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.test_acc(logits.softmax(-1), y)

        self.log(f"loss/xent_{self.protocol}/test", loss, sync_dist=True)
        self.log(f"acc/{self.protocol}/test_epoch", self.test_acc)

        return loss

    def configure_optimizers(self):
        if self.protocol == "linear":
            params = self.linear_layer.parameters()
        elif self.protocol == "finetune":
            params = self.parameters()

        if self.optim == "sdg":
            optimizer = torch.optim.SGD(
                params,
                lr=self.learning_rate,
                nesterov=self.nesterov,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate,
                                         weight_decay=self.weight_decay)

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             self.decay_epochs,
                                                             gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.epochs, eta_min=self.final_lr  # total epochs to run
            )

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # i/o params
        parser.add_argument("--dataset", type=str, default="imagenet",
                            help="dataset")
        parser.add_argument("--data_dir", type=str, default="dataset",
                            help="path to dataset")
        parser.add_argument("--protocol", type=str, default="linear",
                            choices=("linear", "finetune"),
                            help="evalution protocol")
        parser.add_argument("--label_pct", type=int, default=100,
                            help="%% of labels for training")
        parser.add_argument("--ckpt_path", type=str, help="path to ckpt")

        # training params
        parser.add_argument("--num_nodes", default=1, type=int,
                            help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int,
                            help="number of gpus to train on")
        parser.add_argument("--num_workers", default=8, type=int,
                            help="num of workers per GPU")
        parser.add_argument("--max_epochs", default=100, type=int,
                            help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int,
                            help="max steps")
        parser.add_argument("--batch_size", default=256, type=int,
                            help="batch size per gpu")
        parser.add_argument("--fp32", default=True, action=BooleanOptionalAction,
                            help="use fp32 or fp16")
        parser.add_argument("--fast_dev_run", default=False, type=int)

        # fine-tuner params
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--optimizer", type=str, default="sgd")
        parser.add_argument("--learning_rate", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=1e-6)
        parser.add_argument("--label_smoothing", type=float, default=0.0)
        parser.add_argument("--nesterov", type=bool, default=False)
        parser.add_argument("--scheduler_type", type=str, default="cosine")
        parser.add_argument("--gamma", type=float, default=0.1)
        parser.add_argument("--final_lr", type=float, default=0.0)

        return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = PosReconCLREval.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dataset == "imagenet":
        normalization = imagenet_normalization()
        if args.label_pct < 100:
            dm = FewShotImagenetDataModule(args.label_pct,
                                           data_dir=args.data_dir,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers)
        else:
            dm = ImagenetDataModule(data_dir=args.data_dir,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)
        input_height = dm.dims[-1]
    else:
        raise NotImplementedError(f"Unimplemented dataset: {args.dataset}")

    dm.train_transforms = SimCLRFinetuneTransform(
        input_height=input_height,
        normalize=normalization,
        eval_transform=False,
    )

    dm.val_transforms = SimCLRFinetuneTransform(
        input_height=input_height,
        normalize=normalization,
        eval_transform=True,
    )

    dm.test_transforms = SimCLRFinetuneTransform(
        input_height=input_height,
        normalize=normalization,
        eval_transform=True,
    )

    pretrained = PosReconCLR(
        gpus=args.gpus,
        num_samples=1,
        batch_size=args.batch_size,
    ).load_from_checkpoint(args.ckpt_path, strict=False)

    evaluator = PosReconCLREval(
        protocol=args.protocol,
        backbone=pretrained.model,
        in_features=pretrained.embed_dim,
        num_classes=dm.num_classes,
        epochs=args.max_epochs,
        dropout=args.dropout,
        optim=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        nesterov=args.nesterov,
        scheduler_type=args.scheduler_type,
        gamma=args.gamma,
        final_lr=args.final_lr,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        save_last=True,
        monitor=f"acc/{args.protocol}/val_epoch"
    )
    callbacks = [model_checkpoint, lr_monitor]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        devices=args.gpus if args.gpus > 0 else None,
        num_nodes=args.num_nodes,
        accelerator="gpu" if args.gpus > 0 else None,
        strategy="ddp" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(evaluator, datamodule=dm)
    trainer.test(dataloaders=dm)
