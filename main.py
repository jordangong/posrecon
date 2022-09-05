from argparse import ArgumentParser, BooleanOptionalAction
from functools import partial

import torch
from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform, SimCLREvalDataTransform
from pl_bolts.optimizers import linear_warmup_decay, LARS
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from models import MaskedPosReconCLRViT


class PosReconCLR(LightningModule):
    def __init__(
            self,
            gpus: int,
            num_samples: int,
            batch_size: int,
            num_nodes: int = 1,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: int = 4,
            proj_dim: int = 128,
            warmup_epochs: int = 10,
            max_epochs: int = 100,
            shuffle: bool = True,
            mask_ratio: float = 0.75,
            temperature: float = 0.1,
            loss_ratio: float = 1.,
            optimizer: str = "adam",
            exclude_bn_bias: bool = False,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-6,
            **kwargs
    ):
        super(PosReconCLR, self).__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.proj_dim = proj_dim

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.shuffle = shuffle
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.loss_ratio = loss_ratio

        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.model = MaskedPosReconCLRViT(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            proj_dim,
            partial(nn.LayerNorm, eps=1e-6),
        )

        # compute iters per epoch
        global_batch_size = num_nodes * gpus * batch_size if gpus > 0 else batch_size
        self.train_iters_per_epoch = num_samples // global_batch_size

    def shared_step(self, batch):
        (img1, img2, _), _ = batch
        img = torch.cat((img1, img2))
        return self.model(img, self.shuffle, self.mask_ratio, self.temperature)

    def training_step(self, batch, batch_idx):
        latent, pos_embed_pred, proj_embed, loss_recon, loss_clr = self.shared_step(batch)
        loss = self.loss_ratio * loss_recon + loss_clr

        self.log_dict({
            'loss/pretrain/train': loss,
            'loss/recon/train': loss_recon,
            'loss/clr/train': loss_clr,
            'norm/latent': latent.norm(dim=-1).mean(),
            'norm/pos_embed_pred': pos_embed_pred.norm(dim=-1).mean(),
            'norm/proj_embed': proj_embed.norm(dim=-1).mean(),
        }, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        *_, loss_recon, loss_clr = self.shared_step(batch)
        loss = self.loss_ratio * loss_recon + loss_clr

        self.log_dict({
            'loss/pretrain/val': loss,
            'loss/recon/val': loss_recon,
            'loss/clr/val': loss_clr,
        }, sync_dist=True)
        return loss

    @staticmethod
    def exclude_from_wt_decay(named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(),
                                                weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate,
                                         weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument("--img_size", default=224, type=int,
                            help="input image size")
        parser.add_argument("--patch_size", default=16, type=int,
                            help="patch size")
        parser.add_argument("--in_chans", default=3, type=int,
                            help="number of in channels")
        parser.add_argument("--embed_dim", default=768, type=int,
                            help="embedding dimension")
        parser.add_argument("--depth", default=12, type=int,
                            help="number of Transformer blocks")
        parser.add_argument("--num_heads", default=12, type=int,
                            help="number of self-attention heads")
        parser.add_argument("--mlp_ratio", default=4, type=int,
                            help="Ratio of embedding dim to MLP dim")
        parser.add_argument("--proj_dim", default=128, type=int,
                            help="projection head output dimension")
        parser.add_argument("--fp32", default=True, action=BooleanOptionalAction,
                            help="use fp32 or fp16")

        # transform params
        parser.add_argument("--gaussian_blur", default=True,
                            action=BooleanOptionalAction, help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0,
                            help="jitter strength")
        parser.add_argument("--dataset", type=str, default="imagenet",
                            help="dataset")
        parser.add_argument("--data_dir", type=str, default="dataset",
                            help="path to dataset")

        # training params
        parser.add_argument("--fast_dev_run", default=False, type=int)
        parser.add_argument("--num_nodes", default=1, type=int,
                            help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int,
                            help="number of gpus to train on")
        parser.add_argument("--num_workers", default=8, type=int,
                            help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str,
                            help="choose between adam/lars")
        parser.add_argument("--exclude_bn_bias", default=True,
                            action=BooleanOptionalAction,
                            help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=100, type=int,
                            help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int,
                            help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int,
                            help="number of warmup epochs")
        parser.add_argument("--batch_size", default=128, type=int,
                            help="batch size per gpu")

        parser.add_argument("--shuffle", default=True, action=BooleanOptionalAction,
                            help="shuffle patches or not")
        parser.add_argument("--mask_ratio", default=0.75, type=float,
                            help="mask ratio of patches")
        parser.add_argument("--temperature", default=0.1, type=float,
                            help="temperature parameter in InfoNCE loss")
        parser.add_argument("--loss_ratio", default=1., type=float,
                            help="weight of two losses")
        parser.add_argument("--weight_decay", default=1e-6, type=float,
                            help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float,
                            help="base learning rate")

        return parser


if __name__ == '__main__':
    paser = ArgumentParser()
    paser.add_argument("--version", default=None, type=str)
    paser = PosReconCLR.add_model_specific_args(paser)
    args = paser.parse_args()

    if args.dataset == "imagenet":
        normalization = imagenet_normalization()
        dm = ImagenetDataModule(data_dir=args.data_dir,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)
        args.num_samples = dm.num_samples
        args.input_height = dm.dims[-1]
    else:
        raise NotImplementedError(f"Unimplemented dataset: {args.dataset}")

    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    dm.val_transforms = SimCLREvalDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    model = PosReconCLR(**args.__dict__)

    logger = TensorBoardLogger("lightning_logs", name="pretrain", version=args.version)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, monitor="loss/pretrain/val")
    callbacks = [model_checkpoint, lr_monitor]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        devices=args.gpus if args.gpus > 0 else None,
        num_nodes=args.num_nodes,
        accelerator="gpu" if args.gpus > 0 else None,
        strategy="ddp_find_unused_parameters_false" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)
