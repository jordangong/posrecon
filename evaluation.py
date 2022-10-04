from argparse import ArgumentParser, BooleanOptionalAction
from functools import partial

import torch
from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.models.self_supervised import SSLFineTuner
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from timm.data.mixup import Mixup
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from torch import nn
from torchvision import transforms

from main import PosReconCLR
from models import MaskedPosReconCLRViT
from utils.datamodules import FewShotImagenetDataModule
from utils.lr_wt_decay import param_groups_lrd, exclude_from_wt_decay
from utils.transforms import SimCLRFinetuneTransform, imagenet_normalization


class PosReconCLREval(SSLFineTuner):
    def __init__(
            self,
            protocol: str = 'linear',
            dataset: str = 'imagenet',
            img_size: int = 224,
            optim: str = 'sgd',
            exclude_bn_bias: bool = True,
            layer_decay: float = 1.,
            mixup_alpha: float = 0.,
            cutmix_alpha: float = 0.,
            label_smoothing: float = 0.,
            warmup_epochs: int = 10,
            start_lr: float = 0.,
            **kwargs
    ):
        """
        Args:
            protocol: evaluation protocol, including `linear` and `finetune`
            optim: optimizer (SGD or Adam)
            exclude_bn_bias: exclude weight decay on 1d params (e.g. bn/ln and bias)
            mixup_alpha: mixup alpha, active if > 0.
            cutmix_alpha: cutmix alpha, active if > 0.
            label_smoothing: label smoothing regularization
            layer_decay: layer-wise learning rate decay
            warmup_epochs: linear warmup epochs
            start_lr: start learning rate
        """
        assert protocol in {'linear', 'finetune'}, f"unknown protocol: {protocol}"
        assert optim in {'sgd', 'adam'}, f"unknown optimizer: {optim}"

        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=['backbone'])

        self.protocol = protocol
        self.dataset = dataset
        self.optim = optim
        self.exclude_bn_bias = exclude_bn_bias
        self.mixup = None
        if mixup_alpha > 0. or cutmix_alpha > 0.:
            self.mixup = Mixup(mixup_alpha, cutmix_alpha,
                               label_smoothing=label_smoothing,
                               num_classes=self.linear_layer.n_classes)
        self.label_smoothing = label_smoothing
        self.layer_decay = layer_decay
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr

        if self.mixup is None:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = SoftTargetCrossEntropy()

        if dataset == "imagenet":
            normalization = imagenet_normalization()
        self.train_transform = SimCLRFinetuneTransform(
            img_size=img_size,
            normalize=normalization,
            eval_transform=False,
        )
        self.eval_transform = SimCLRFinetuneTransform(
            img_size=img_size,
            normalize=normalization,
            eval_transform=True,
        )

    def on_train_epoch_start(self) -> None:
        if self.protocol == 'linear':
            self.backbone.eval()
        elif self.protocol == 'finetune':
            self.backbone.train()

    def forward_resnet(self, x):
        return self.backbone.encoder(x)[0]

    def forward_vit(self, x):
        x = self.backbone.patch_embed(x)
        x += self.backbone.pos_embed

        cls_tokens = self.backbone.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        return x[:, 0, :]

    def forward_backbone(self, x):
        if isinstance(self.backbone, MaskedPosReconCLRViT):
            return self.forward_vit(x)
        else:
            return self.forward_resnet(x)

    def shared_step(self, batch):
        x, y = batch
        if self.protocol == 'linear':
            with torch.no_grad():
                feats = self.forward_backbone(x)
        elif self.protocol == 'finetune':
            feats = self.forward_backbone(x)

        logits = self.linear_layer(feats)
        loss = self.criterion(logits, y)

        return loss, logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.train_transform(x)
        target = y.clone()
        if self.mixup is not None:
            x, y = self.mixup(x, y)
        loss, logits = self.shared_step((x, y))
        acc = self.train_acc(logits.softmax(-1), target)

        self.log(f"loss/xent_{self.protocol}/train", loss, prog_bar=True)
        self.log(f"acc/{self.protocol}/train_step", acc, prog_bar=True)
        self.log(f"acc/{self.protocol}/train_epoch", self.train_acc,
                 on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.eval_transform(x)
        loss, logits = self.shared_step((x, y))
        self.val_acc(logits.softmax(-1), y)

        self.log(f"loss/xent_{self.protocol}/val",
                 loss, prog_bar=True, sync_dist=True)
        self.log(f"acc/{self.protocol}/val_epoch", self.val_acc)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.eval_transform(x)
        loss, logits = self.shared_step((x, y))
        self.test_acc(logits.softmax(-1), y)

        self.log(f"loss/xent_{self.protocol}/test", loss, sync_dist=True)
        self.log(f"acc/{self.protocol}/test_epoch", self.test_acc)

        return loss

    def configure_optimizers(self):
        param_groups = []
        # add backbone to params_groups while finetuning
        if self.protocol == "finetune":
            if isinstance(self.backbone, MaskedPosReconCLRViT):
                param_groups += param_groups_lrd(
                    self.backbone,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    exclude_1d_params=self.exclude_bn_bias,
                    no_weight_decay_list=("pos_embed", "cls_token"),
                    layer_decay=self.layer_decay
                )
            else:  # ResNet
                if self.exclude_bn_bias:
                    resnet_param_groups = exclude_from_wt_decay(
                        self.backbone,
                        self.weight_decay,
                    )
                else:
                    resnet_param_groups = [{
                        "params": self.backbone.parameters(),
                        "weight_decay": self.weight_decay
                    }]
                param_groups += resnet_param_groups

        # add linear head
        if self.exclude_bn_bias:
            linear_param_groups = exclude_from_wt_decay(
                self.linear_layer,
                self.weight_decay,
            )
        else:
            linear_param_groups = [{
                "params": self.linear_layer.parameters(),
                "weight_decay": self.weight_decay
            }]
        param_groups += linear_param_groups

        if self.optim == "sdg":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.learning_rate,
                nesterov=self.nesterov,
                momentum=0.9,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=self.learning_rate)

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             self.decay_epochs,
                                                             gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.epochs, eta_min=self.final_lr  # total epochs to run
            )
        elif self.scheduler_type == "warmup-anneal":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, self.warmup_epochs, max_epochs=self.epochs,
                warmup_start_lr=self.start_lr, eta_min=self.final_lr
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
        parser.add_argument("--warmup_epochs", default=10, type=int,
                            help="number of warmup epochs")
        parser.add_argument("--fp32", default=True, action=BooleanOptionalAction,
                            help="use fp32 or fp16")
        parser.add_argument("--fast_dev_run", default=False, type=int)

        # fine-tuner params
        parser.add_argument("--mlp_dropout", type=float, default=0.0)
        parser.add_argument("--attention_dropout", type=float, default=0.0)
        parser.add_argument("--path_dropout", type=float, default=0.0)
        parser.add_argument("--head_dropout", type=float, default=0.0)
        parser.add_argument("--optimizer", type=str, default="sgd")
        parser.add_argument("--exclude_bn_bias", default=True, action=BooleanOptionalAction)
        parser.add_argument("--learning_rate", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=1e-6)
        parser.add_argument("--layer_decay", type=float, default=1.0)
        parser.add_argument("--mixup_alpha", type=float, default=0.0)
        parser.add_argument("--cutmix_alpha", type=float, default=0.0)
        parser.add_argument("--label_smoothing", type=float, default=0.0)
        parser.add_argument("--nesterov", type=bool, default=False)
        parser.add_argument("--scheduler_type", type=str, default="warmup-anneal")
        parser.add_argument("--gamma", type=float, default=0.1)
        parser.add_argument("--start_lr", type=float, default=1e-6)
        parser.add_argument("--final_lr", type=float, default=1e-6)

        return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", default=None, type=str)
    parser.add_argument("--log_path", default="lightning_logs", type=str)
    parser.add_argument("--resume_ckpt_path", default=None, type=str)
    parser = PosReconCLREval.add_model_specific_args(parser)
    args = parser.parse_args()

    pretrained = PosReconCLR.load_from_checkpoint(args.ckpt_path, strict=False)
    # a bit hacky here, replace ViT with dropout rate
    if isinstance(pretrained.model, MaskedPosReconCLRViT):
        pretained_state_dict = pretrained.state_dict()
        pretrained.model = MaskedPosReconCLRViT(
            pretrained.img_size,
            pretrained.patch_size,
            pretrained.in_chans,
            pretrained.embed_dim,
            pretrained.encoder_depth,
            pretrained.encoder_num_heads,
            pretrained.decoder_depth,
            pretrained.decoder_num_heads,
            pretrained.mlp_ratio,
            pretrained.proj_dim,
            drop_rate=args.mlp_dropout,
            attention_drop_rate=args.attention_dropout,
            drop_path_rate=args.path_dropout,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        pretrained.load_state_dict(pretained_state_dict)

    if args.dataset == "imagenet":
        if args.label_pct < 100:
            dm = FewShotImagenetDataModule(args.label_pct,
                                           data_dir=args.data_dir,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers)
        else:
            dm = ImagenetDataModule(data_dir=args.data_dir,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)
    else:
        raise NotImplementedError(f"Unimplemented dataset: {args.dataset}")

    dm.train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(pretrained.img_size),
        transforms.ToTensor(),
    ])
    dm.val_transforms = dm.test_transforms = transforms.Compose([
        transforms.Resize(int(pretrained.img_size + 0.1 * pretrained.img_size)),
        transforms.CenterCrop(pretrained.img_size),
        transforms.ToTensor(),
    ])

    evaluator = PosReconCLREval(
        protocol=args.protocol,
        dataset=args.dataset,
        img_size=pretrained.img_size,
        backbone=pretrained.model,
        in_features=pretrained.embed_dim,
        num_classes=dm.num_classes,
        epochs=args.max_epochs,
        dropout=args.head_dropout,
        optim=args.optimizer,
        exclude_bn_bias=args.exclude_bn_bias,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        layer_decay=args.layer_decay,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        label_smoothing=args.label_smoothing,
        nesterov=args.nesterov,
        scheduler_type=args.scheduler_type,
        gamma=args.gamma,
        warmup_epochs=args.warmup_epochs,
        start_lr=args.start_lr,
        final_lr=args.final_lr,
    )

    logger = TensorBoardLogger(args.log_path, name="evaluation", version=args.version)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        save_last=True,
        monitor=f"loss/xent_{args.protocol}/val"
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
        logger=logger,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(evaluator, datamodule=dm, ckpt_path=args.resume_ckpt_path)
    trainer.test(dataloaders=dm, ckpt_path='last')
