import copy
from argparse import ArgumentParser, BooleanOptionalAction
from functools import partial

import torch
import torch.nn.functional as F
import torch_optimizer as optim
from pl_bolts.callbacks.knn_online import KNNOnlineEvaluator
from pl_bolts.datamodules import ImagenetDataModule
from pl_bolts.optimizers import linear_warmup_decay
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from models import SimCLRViT, SyncFunction
from utils.datamodules import FewShotImagenetDataModule
from utils.lr_wt_decay import param_groups_lrd
from utils.transforms import SimCLRPretrainPostTransform, imagenet_normalization, SimCLRPretrainPreTransform


class MuitiHeadAttnMaskCLR(LightningModule):
    def __init__(
            self,
            gpus: int,
            num_samples: int,
            batch_size: int,
            num_nodes: int = 1,
            dataset: str = "imagenet",
            gaussian_blur: bool = True,
            jitter_strength: float = 1.,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: int = 4,
            proj_dim: int = 128,
            drop_rate: float = 0.,
            attention_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            warmup_epochs: int = 10,
            max_epochs: int = 100,
            position: bool = True,
            online_mask_ratio: float = 0.75,
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
        super(MuitiHeadAttnMaskCLR, self).__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.dataset = dataset
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.num_samples = num_samples
        self.batch_size = batch_size

        # ViT params
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
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
        self.online_mask_ratio = online_mask_ratio
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

        self.online_net = SimCLRViT(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            proj_dim,
            drop_rate,
            attention_drop_rate,
            drop_path_rate,
            partial(nn.LayerNorm, eps=1e-6),
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
        x, *_ = self.online_net(x, position)

        return x[:, 0, :]

    @staticmethod
    def multi_head_attn_mask(patch_embed, attn_weight, mask_ratio):
        """
        Multi-head attention-guided masking
        Args:
            patch_embed: [batch_size * (num_heads + 1), seq_len, embed_dim]
            attn_weight: [batch_size, num_heads, 1 + seq_len, 1 + seq_len]
            mask_ratio: ratio of # of masked patches to # of patches
        return:
            masked_patch_embed: [batch_size * (num_heads + 1),
                                 seq_len * (1 - mask_ratio), embed_dim]
        """
        _, seq_len, embed_dim = patch_embed.size()

        cls_attn_weight = attn_weight[:, :, 0, 1:]
        # cls_attn_weight: [batch_size, num_heads, seq_len]
        cls_attn_head_avg_weight = cls_attn_weight.mean(1)
        # cls_attn_head_avg_weight: [batch_size, seq_len]
        cls_attn_weight = torch.cat((cls_attn_weight.view(-1, seq_len),
                                     cls_attn_head_avg_weight))
        # cls_attn_weight: [batch_size * (num_heads + 1), seq_len]
        attn_ranked_indices = cls_attn_weight.argsort(descending=True)

        visible_len = int(seq_len * (1 - mask_ratio))
        visible_indices = attn_ranked_indices[:, :visible_len]
        # visible_indices: [batch_size * (num_heads + 1), seq_len * mask_ratio]
        expand_visible_indices = visible_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        # expand_visible_indices: [batch_size * (num_heads + 1), seq_len * (1 - mask_ratio), embed_dim]
        masked_patch_embed = patch_embed.gather(1, expand_visible_indices)
        # masked_patch_embed: [batch_size * (num_heads + 1), seq_len * (1 - mask_ratio), embed_dim]

        return masked_patch_embed

    @staticmethod
    def instance_contrast_loss(proj_online, proj_target, temp):
        # Instance-level contrast
        # positive pairs: mean attention masked and target
        # proj_online: [batch_size, num_heads + 1, proj_dim]
        # proj_target: [batch_size, proj_dim]
        feat = torch.cat((proj_online, proj_target.unsqueeze(1)), dim=1)
        # feat: [batch_size, num_heads + 2, proj_dim]
        all_sim = (feat @ feat.transpose(1, 2))
        all_sim.diagonal(dim1=-2, dim2=-1).fill_(0)
        # all_sim: [batch_size, num_heads + 2, num_heads + 2]
        all_ = torch.exp(all_sim / temp).sum(dim=(-2, -1))
        # all_: [batch_size]
        pos_sim = (proj_online[:, -1:] @ proj_target.unsqueeze(-1)).squeeze()
        # pos_sim: [batch_size]
        pos = torch.exp(pos_sim / temp) * 2
        loss = -torch.log(pos / all_).mean()

        return loss, all_sim

    @staticmethod
    def batch_contrast_loss(proj_online, proj_target, sim_instance, temp):
        # Batch-level contrast
        # positive pairs: all instance-level pairs
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            proj_online = SyncFunction.apply(proj_online)
            proj_target = SyncFunction.apply(proj_target)
            sim_instance = SyncFunction.apply(sim_instance)
        # proj_online: [batch_size*2 (* world_size), num_heads + 1, proj_dim]
        # proj_target: [batch_size*2 (* world_size), proj_dim]
        # sim_instance: [batch_size*2 (* world_size), num_heads + 2, num_heads + 2]
        *_, proj_dim = proj_online.size()
        feat = torch.cat((proj_online.view(-1, proj_dim), proj_target))
        # feat: [batch_size*2 (* world_size) * (num_heads+2), proj_dim]
        all_sim = (feat @ feat.T).fill_diagonal_(0)
        # feat: [batch_size*2 (* world_size) * (num_heads+2), ...]
        all_ = torch.exp(all_sim / temp).sum(-1)
        pos = torch.exp(sim_instance / temp).sum(dim=-1).view(-1)
        loss = -torch.log(pos / all_).mean()

        return loss

    def shared_step(self, batch):
        (crop1, crop2), _ = batch
        batch_size, *img_dim = crop1.size()
        crops = torch.stack((crop1, crop2))
        # crops: [2, batch_size, in_chans, height, width]
        num_masks = self.num_heads + 1  # Last one for attention mean
        #                                             another one for target
        crops = crops.unsqueeze(2).expand(-1, -1, num_masks + 1, *img_dim)
        # crops: [2, batch_size, num_heads+2, in_chans, height, width]
        img = self.transform(crops.reshape(-1, *img_dim)).view_as(crops)
        # img: [2, batch_size, num_heads+2, in_chans, height, width]

        # To fetch attention weight, forward target net first
        img_target = img[:, :, 0].reshape(-1, *img_dim)
        with torch.no_grad():
            _, proj_target_, attn_weight = self.target_net(img_target, self.position)

        # To mask on embedding-level, manually forward online encoder
        img_online = img[:, :, 1:].reshape(-1, *img_dim)
        patch_embed = self.online_net.pre_encode(img_online, self.position)
        masked_patch_embed = self.multi_head_attn_mask(
            patch_embed, attn_weight, self.online_mask_ratio
        )
        latent, _ = self.online_net.forward_encoder(masked_patch_embed)
        proj_online_ = self.online_net.proj_head(latent[:, 0, :])

        # Normalize and reshape for loss calculation
        proj_target, proj_online = F.normalize(proj_target_), F.normalize(proj_online_)
        proj_online = proj_online.view(-1, self.num_heads + 1, self.proj_dim)
        proj1_online, proj2_online = proj_online.chunk(2)
        proj1_target, proj2_target = proj_target.chunk(2)

        # Instance-level loss
        loss_instance_clr1, sim_instance1 = self.instance_contrast_loss(
            proj1_online, proj2_target, self.temperature
        )
        loss_instance_clr2, sim_instance2 = self.instance_contrast_loss(
            proj2_online, proj1_target, self.temperature
        )
        loss_instance_clr = (loss_instance_clr1 + loss_instance_clr2) / 2

        # Batch-level loss
        loss_batch_clr1 = self.batch_contrast_loss(
            proj1_online, proj2_target, sim_instance1, self.temperature
        )
        loss_batch_clr2 = self.batch_contrast_loss(
            proj2_online, proj1_target, sim_instance2, self.temperature
        )
        loss_batch_clr = (loss_batch_clr1 + loss_batch_clr2) / 2

        return latent, (proj_online_, proj_target_), (loss_instance_clr, loss_batch_clr)

    def training_step(self, batch, batch_idx):
        latent, proj_embed, losses = self.shared_step(batch)
        loss = self.loss_ratio * losses[0] + losses[1]

        self.log_dict({
            'loss/pretrain/train': loss,
            'loss/instance/train': losses[0],
            'loss/batch/train': losses[1],
            'norm/latent': latent.norm(dim=-1).mean(),
            'norm/proj_embed_online': proj_embed[0].norm(dim=-1).mean(),
            'norm/proj_embed_target': proj_embed[1].norm(dim=-1).mean(),
        }, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        *_, losses = self.shared_step(batch)
        loss = self.loss_ratio * losses[0] + losses[1]

        self.log_dict({
            'loss/pretrain/val': loss,
            'loss/instance/val': losses[0],
            'loss/batch/val': losses[1],
        }, sync_dist=True)
        return loss

    def configure_optimizers(self):
        param_groups = param_groups_lrd(
            self.online_net,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            exclude_1d_params=self.exclude_bn_bias,
            no_weight_decay_list=("pos_embed", "cls_token"),
            layer_decay=self.layer_decay
        )

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
        parser.add_argument("--online_mask_ratio", default=0.75, type=float,
                            help="online network mask ratio of patches")
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
    parser = MuitiHeadAttnMaskCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dataset == "imagenet":
        if args.sample_pct < 100:
            dm = FewShotImagenetDataModule(args.data_dir,
                                           label_pct=args.sample_pct,
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

    model = MuitiHeadAttnMaskCLR(**args.__dict__)

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
