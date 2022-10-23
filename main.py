import copy
from argparse import ArgumentParser, BooleanOptionalAction
from functools import partial

import torch
import torch_optimizer as optim
from pl_bolts.callbacks.knn_online import KNNOnlineEvaluator
from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.optimizers import linear_warmup_decay
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from models import MaskedPosReconCLRViT, SimCLRResNet, info_nce_loss
from utils.datamodules import FewShotImagenetDataModule
from utils.lr_wt_decay import param_groups_lrd, exclude_from_wt_decay
from utils.transforms import SimCLRPretrainPostTransform, imagenet_normalization, SimCLRPretrainPreTransform


class PosReconCLR(LightningModule):
    def __init__(
            self,
            gpus: int,
            num_samples: int,
            batch_size: int,
            num_nodes: int = 1,
            dataset: str = "imagenet",
            backbone: str = "vit",
            layers: tuple = (3, 4, 6, 3),
            gaussian_blur: bool = True,
            jitter_strength: float = 1.,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            encoder_depth: int = 12,
            encoder_num_heads: int = 12,
            decoder_depth: int = 0,
            decoder_num_heads: int = 12,
            mlp_ratio: int = 4,
            proj_dim: int = 128,
            drop_rate: float = 0.,
            attention_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            warmup_epochs: int = 10,
            max_epochs: int = 100,
            position: bool = True,
            shuffle: bool = True,
            online_mask_ratio: float = 0.75,
            target_mask_ratio: float = 0.,
            attn_mask: bool = False,
            attn_mask_mode: str = 'high',
            hint_ratio: float = 0.,
            temperature: float = 0.1,
            loss_ratio: float = 1.,
            optimizer: str = "adam",
            exclude_bn_bias: bool = False,
            learning_rate: float = 1e-3,
            ema_momentum: float = 0.99,
            weight_decay: float = 1e-6,
            layer_decay: float = 1.,
            **kwargs
    ):
        super(PosReconCLR, self).__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.dataset = dataset
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.backbone = backbone

        # ResNet params
        self.layers = layers

        # ViT params
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.proj_dim = proj_dim
        self.drop_rate = drop_rate
        self.attention_drop_rate = attention_drop_rate
        self.drop_path_rate = drop_path_rate

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.layer_decay = layer_decay
        self.position = position
        self.shuffle = shuffle
        self.online_mask_ratio = online_mask_ratio
        self.target_mask_ratio = target_mask_ratio
        self.attn_mask = attn_mask
        self.attn_mask_mode = attn_mask_mode
        self.hint_ratio = hint_ratio
        self.temperature = temperature
        self.loss_ratio = loss_ratio

        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.ema_momentum = ema_momentum

        if dataset == "imagenet":
            normalization = imagenet_normalization()
        self.normalization = normalization
        self.transform = SimCLRPretrainPostTransform(
            img_size=img_size,
            gaussian_blur=gaussian_blur,
            jitter_strength=jitter_strength,
            normalize=normalization,
        )

        self.online_net = MaskedPosReconCLRViT(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            encoder_depth,
            encoder_num_heads,
            decoder_depth,
            decoder_num_heads,
            mlp_ratio,
            proj_dim,
            drop_rate,
            attention_drop_rate,
            drop_path_rate,
            partial(nn.LayerNorm, eps=1e-6),
        ) if backbone == "vit" else SimCLRResNet(
            layers=layers,
            embed_dim=embed_dim,
            proj_dim=proj_dim,
        )
        self.target_net = copy.deepcopy(self.online_net)

        # compute iters per epoch
        global_batch_size = num_nodes * gpus * batch_size if gpus > 0 else batch_size
        self.train_iters_per_epoch = num_samples // global_batch_size

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for online_p, target_p in zip(self.online_net.parameters(),
                                      self.target_net.parameters()):
            em = self.ema_momentum
            target_p.data = target_p.data * em + online_p.data * (1.0 - em)

    def forward(self, x, position=True):
        x = self.normalization(x)
        x = self.online_net(x, position, pretrain=False)

        return x

    def shared_step(self, batch):
        img, _ = batch
        img = self.transform(torch.cat(img))

        latent, pos_pred, vis_ids, proj_online = self.online_net(
            img, self.position, self.shuffle, self.online_mask_ratio,
            self.attn_mask, self.attn_mask_mode, self.hint_ratio
        )
        with torch.no_grad():
            proj_target = self.target_net(
                img, self.position, False, self.target_mask_ratio,
                self.attn_mask, self.attn_mask_mode, self.hint_ratio,
                target_network=True,
            )

        loss_recon = self.online_net.pos_recon_loss(pos_pred, vis_ids)

        proj1_online, proj2_online = proj_online.chunk(2)
        proj1_target, proj2_target = proj_target.chunk(2)
        loss_clr1 = info_nce_loss(proj1_online, proj2_target, self.temperature)
        loss_clr2 = info_nce_loss(proj2_online, proj1_target, self.temperature)
        loss_clr = (loss_clr1 + loss_clr2) / 2

        return latent, pos_pred, (proj_online, proj_target), loss_recon, loss_clr

    def training_step(self, batch, batch_idx):
        latent, pos_embed_pred, proj_embed, loss_recon, loss_clr = self.shared_step(batch)
        loss = self.loss_ratio * loss_recon + loss_clr

        self.log_dict({
            'loss/pretrain/train': loss,
            'loss/recon/train': loss_recon,
            'loss/clr/train': loss_clr,
            'norm/latent': latent.norm(dim=-1).mean(),
            'norm/pos_embed_pred': pos_embed_pred.norm(dim=-1).mean(),
            'norm/proj_embed_online': proj_embed[0].norm(dim=-1).mean(),
            'norm/proj_embed_target': proj_embed[1].norm(dim=-1).mean(),
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

    def configure_optimizers(self):
        if self.backbone == "vit":
            param_groups = param_groups_lrd(
                self.online_net,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                exclude_1d_params=self.exclude_bn_bias,
                no_weight_decay_list=("pos_embed", "cls_token"),
                layer_decay=self.layer_decay
            )
        else:
            if self.exclude_bn_bias:
                param_groups = exclude_from_wt_decay(
                    self.online_net,
                    self.weight_decay
                )
            else:
                param_groups = {
                    "params": self.online_net.parameters(),
                    "weight_decay": self.weight_decay
                }

        if self.optim == "lars":
            optimizer = optim.LARS(param_groups, lr=self.learning_rate, momentum=0.9)
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=self.learning_rate)
        elif self.optim == "lamb":
            optimizer = optim.Lamb(param_groups, lr=self.learning_rate)
        elif self.optim == "adamw":
            optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)

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
        parser.add_argument("--backbone", default="vit", type=str,
                            help="backbone (ViT or ResNet)")
        parser.add_argument("--layers", default=(3, 4, 6, 3), nargs=4, type=int,
                            help="number of ResNet layers in each block")
        parser.add_argument("--img_size", default=224, type=int,
                            help="input image size")
        parser.add_argument("--patch_size", default=16, type=int,
                            help="patch size")
        parser.add_argument("--in_chans", default=3, type=int,
                            help="number of in channels")
        parser.add_argument("--embed_dim", default=768, type=int,
                            help="embedding dimension")
        parser.add_argument("--encoder_depth", default=12, type=int,
                            help="encoder number of Transformer blocks")
        parser.add_argument("--encoder_num_heads", default=12, type=int,
                            help="encoder number of self-attention heads")
        parser.add_argument("--decoder_depth", default=0, type=int,
                            help="decoder number of Transformer blocks, "
                                 "set it to 0 for linear layer")
        parser.add_argument("--decoder_num_heads", default=12, type=int,
                            help="decoder number of self-attention heads")
        parser.add_argument("--mlp_ratio", default=4, type=int,
                            help="Ratio of embedding dim to MLP dim")
        parser.add_argument("--proj_dim", default=128, type=int,
                            help="projection head output dimension")
        parser.add_argument("--mlp_dropout", default=0.0, type=float,
                            help="mlp dropout rate")
        parser.add_argument("--attention_dropout", default=0.0, type=float,
                            help="attention dropout rate")
        parser.add_argument("--path_dropout", default=0.0, type=float,
                            help="path dropout rate")
        parser.add_argument("--fp32", default=True, action=BooleanOptionalAction,
                            help="use fp32 or fp16")

        # transform params
        parser.add_argument("--dataset", type=str, default="imagenet",
                            help="dataset")
        parser.add_argument("--data_dir", type=str, default="dataset",
                            help="path to dataset")
        parser.add_argument("--sample_pct", type=int, default=100,
                            help="%% of samples for training (only for ablation)")
        parser.add_argument("--gaussian_blur", default=True,
                            action=BooleanOptionalAction, help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0,
                            help="jitter strength")

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
                            help="exclude bn/ln/bias from weight decay")
        parser.add_argument("--max_epochs", default=100, type=int,
                            help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int,
                            help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int,
                            help="number of warmup epochs")
        parser.add_argument("--ema_momentum", default=0.99, type=float,
                            help="ema momentum")
        parser.add_argument("--batch_size", default=128, type=int,
                            help="batch size per gpu")

        parser.add_argument("--position", default=True, action=BooleanOptionalAction,
                            help="add positional embedding or not")
        parser.add_argument("--shuffle", default=True, action=BooleanOptionalAction,
                            help="shuffle patches or not")
        parser.add_argument("--online_mask_ratio", default=0.75, type=float,
                            help="online network mask ratio of patches")
        parser.add_argument("--target_mask_ratio", default=0., type=float,
                            help="target network mask ratio of patches")
        parser.add_argument("--attn_mask", default=False, action='store_true',
                            help="mask patches guided by attention")
        parser.add_argument("--attn_mask_mode", default='high', type=str,
                            help="attn-mask mode: "
                                 "high for masking out high attention patches, "
                                 "low for low attention patches")
        parser.add_argument("--hint_ratio", default=0., type=float,
                            help="attention hint ratio "
                                 "(leave some high attention patches)")
        parser.add_argument("--temperature", default=0.1, type=float,
                            help="temperature parameter in InfoNCE loss")
        parser.add_argument("--loss_ratio", default=1., type=float,
                            help="weight of two losses")
        parser.add_argument("--weight_decay", default=1e-6, type=float,
                            help="weight decay")
        parser.add_argument("--layer_decay", default=1., type=float,
                            help="layer-wise decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float,
                            help="base learning rate")

        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--version", default=None, type=str)
    parser.add_argument("--log_path", default="lightning_logs", type=str)
    parser.add_argument("--resume_ckpt_path", default=None, type=str)
    parser.add_argument("--track_grad", default=True, type=BooleanOptionalAction)
    parser.add_argument("--knn_probe", default=True, type=BooleanOptionalAction)
    parser = PosReconCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dataset == "imagenet":
        if args.sample_pct < 100:
            dm = FewShotImagenetDataModule(args.data_dir,
                                           label_pct=args.sample_pct,
                                           num_imgs_per_val_class=1,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers)
        else:
            dm = ImagenetDataModule(data_dir=args.data_dir,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)
        args.num_samples = dm.num_samples
    else:
        raise NotImplementedError(f"Unimplemented dataset: {args.dataset}")

    dm.train_transforms = dm.val_transforms = SimCLRPretrainPreTransform(args.img_size)

    model = PosReconCLR(**args.__dict__)

    logger = TensorBoardLogger(args.log_path, name="pretrain", version=args.version)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, monitor="loss/pretrain/val")

    callbacks = [model_checkpoint]
    if args.knn_probe:
        callbacks.append(KNNOnlineEvaluator())
    callbacks.append(lr_monitor)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        devices=args.gpus if args.gpus > 0 else None,
        num_nodes=args.num_nodes,
        accelerator="gpu" if args.gpus > 0 else None,
        strategy="ddp_find_unused_parameters_false" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        track_grad_norm=2 if args.track_grad else -1,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt_path)
