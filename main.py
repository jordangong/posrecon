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
from torchmetrics import Accuracy

from models import SyncFunction, SimCLRMaskedViT
from utils.datamodules import FewShotImagenetDataModule
from utils.lr_wt_decay import param_groups_lrd, exclude_from_wt_decay
from utils.transforms import SimCLRPretrainPostTransform, imagenet_normalization, SimCLRPretrainPreTransform


class MultiRandMaskCLR(LightningModule):
    def __init__(
            self,
            gpus: int,
            num_samples: int,
            batch_size: int,
            num_nodes: int = 1,
            dataset: str = "imagenet",
            num_classes: int = 1000,
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
            mask_ratio: float = 0.75,
            num_masks: int = 1,
            temperature: float = 0.1,
            optimizer: str = "adam",
            exclude_bn_bias: bool = False,
            learning_rate: float = 1e-3,
            ema_momentum: float = 0.,
            weight_decay: float = 1e-6,
            layer_decay: float = 1.,
            **kwargs
    ):
        super(MultiRandMaskCLR, self).__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.dataset = dataset
        self.num_classes = num_classes
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
        self.mask_ratio = mask_ratio
        self.num_masks = num_masks
        self.temperature = temperature

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

        self.siamese_net = SimCLRMaskedViT(
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
        if ema_momentum > 0:
            self.siamese_net_sg = copy.deepcopy(self.siamese_net)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.train_acc_top_1 = Accuracy(top_k=1)
        self.train_acc_top_5 = Accuracy(top_k=5)
        self.val_acc_top_1 = Accuracy(top_k=1, compute_on_step=False)
        self.val_acc_top_5 = Accuracy(top_k=5, compute_on_step=False)

        # compute iters per epoch
        global_batch_size = num_nodes * gpus * batch_size if gpus > 0 else batch_size
        self.train_iters_per_epoch = num_samples // global_batch_size

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_momentum > 0:
            for online_p, target_p in zip(self.siamese_net.parameters(),
                                          self.siamese_net_sg.parameters()):
                em = self.ema_momentum
                target_p.data = target_p.data * em + online_p.data * (1.0 - em)

    def forward(self, x, position=True):
        x = self.normalization(x)
        x, *_ = self.siamese_net(x, position)

        return x

    @staticmethod
    def batch_contrast_loss(proj, temp):
        # Batch-level contrast (all views), similar to InfoNCE but with more positive pairs
        # swap batch size to the first, make it easier for all_gather
        proj = proj.transpose(0, 1).contiguous()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            proj = SyncFunction.apply(proj)
        # proj: [batch_size (* world_size), num_crops * num_masks, proj_dim]
        batch_size, num_pos_views, proj_dim = proj.size()

        pos_sim = proj @ proj.transpose(1, 2)
        pos_sim = pos_sim.masked_select(~torch.eye(
            num_pos_views, dtype=torch.bool, device=proj.device
        ))
        pos_sim = pos_sim.view(batch_size * num_pos_views, -1)
        pos = torch.exp(pos_sim / temp)

        neg_sim = []
        for pos_idx in torch.eye(batch_size, dtype=torch.bool):
            neg_sim.append(proj[pos_idx].squeeze(0)
                           @ proj[~pos_idx].view(-1, proj_dim).T)
        neg_sim = torch.stack(neg_sim).view(batch_size * num_pos_views, -1)
        neg = torch.exp(neg_sim / temp).sum(-1)

        all_ = pos + neg.unsqueeze(-1)

        loss = -torch.log(pos / all_).mean()

        return loss

    def shared_step(self, img):
        crop1, crop2 = img
        batch_size, *img_dim = crop1.size()
        crops = torch.stack((crop1, crop2))
        # crops: [2, batch_size, in_chans, height, width]
        crops = crops.unsqueeze(1).expand(-1, self.num_masks, -1, *img_dim)
        # crops: [2, num_masks, batch_size, in_chans, height, width]
        img = self.transform(crops.reshape(-1, *img_dim))
        # img: [2 * num_masks * batch_size, in_chans, height, width]

        reps, proj_, _ = self.siamese_net(img, self.position, self.mask_ratio)
        proj = F.normalize(proj_).view(-1, batch_size, self.proj_dim)
        if self.ema_momentum > 0:
            with torch.no_grad():
                _, proj_sg_, _ = self.siamese_net_sg(img, self.position, self.mask_ratio)

            proj1, proj2 = proj.chunk(2)
            proj_sg = F.normalize(proj_sg_).view(-1, batch_size, self.proj_dim)
            proj1_sg, proj2_sg = proj_sg.chunk(2)
            proj_1to2 = torch.cat((proj1, proj2_sg))
            proj_2to1 = torch.cat((proj2, proj1_sg))
            loss_batch_clr_a = self.batch_contrast_loss(proj_1to2, self.temperature)
            loss_batch_clr_b = self.batch_contrast_loss(proj_2to1, self.temperature)
            loss_batch_clr = (loss_batch_clr_a + loss_batch_clr_b) / 2
        else:
            loss_batch_clr = self.batch_contrast_loss(proj, self.temperature)

        return reps, proj_, loss_batch_clr

    def linear_probe(self, reps, labels, acc_top_1_fn, acc_top_5_fn):
        logits = self.classifier(reps)
        loss = F.cross_entropy(logits, labels)
        acc_top_1 = acc_top_1_fn(logits.softmax(-1), labels)
        acc_top_5 = acc_top_5_fn(logits.softmax(-1), labels)

        return loss, acc_top_1, acc_top_5

    def training_step(self, batch, batch_idx):
        img, labels = batch
        reps, proj, loss = self.shared_step(img)

        num_crops = len(img)
        label_expand = labels.unsqueeze(0).expand(num_crops * self.num_masks, -1).reshape(-1)
        loss_xent, acc_top_1, acc_top_5 = self.linear_probe(
            reps.detach(), label_expand, self.train_acc_top_1, self.train_acc_top_5
        )

        self.log_dict({
            'loss/pretrain/train': loss,
            'loss/linear_probe/train': loss_xent,
            'acc/linear_probe_top_1/train': acc_top_1,
            'acc/linear_probe_top_5/train': acc_top_5,
            'norm/reps': reps.norm(dim=-1).mean(),
            'norm/proj': proj.norm(dim=-1).mean(),
        }, sync_dist=True)
        return loss + loss_xent

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        reps, _, loss = self.shared_step(img)

        num_crops = len(img)
        label_expand = labels.unsqueeze(0).expand(num_crops * self.num_masks, -1).reshape(-1)
        loss_xent, acc_top_1, acc_top_5 = self.linear_probe(
            reps.detach(), label_expand, self.val_acc_top_1, self.val_acc_top_5
        )

        self.log_dict({
            'loss/pretrain/val': loss,
            'loss/linear_probe/val': loss_xent,
            'acc/linear_probe_top_1/val': acc_top_1,
            'acc/linear_probe_top_5/val': acc_top_5,
        }, sync_dist=True)
        return loss + loss_xent

    def configure_optimizers(self):
        param_groups = param_groups_lrd(
            self.siamese_net,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            exclude_1d_params=self.exclude_bn_bias,
            no_weight_decay_list=("pos_embed", "cls_token"),
            layer_decay=self.layer_decay
        )
        if self.exclude_bn_bias:
            param_groups += exclude_from_wt_decay(self.classifier, self.weight_decay)
        else:
            param_groups += [
                {"params": self.classifier.parameters(), "weight_decay": self.weight_decay}
            ]

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
        parser.add_argument("--ema_momentum", default=0., type=float,
                            help="ema momentum")
        parser.add_argument("--batch_size", default=128, type=int,
                            help="batch size per gpu")

        parser.add_argument("--position", default=True, action=BooleanOptionalAction,
                            help="add positional embedding or not")
        parser.add_argument("--mask_ratio", default=0.75, type=float,
                            help="mask ratio of patches")
        parser.add_argument("--num_masks", default=1, type=int,
                            help="number of masks per crop")
        parser.add_argument("--temperature", default=0.1, type=float,
                            help=" temperature in loss")
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
    parser.add_argument("--track_grad", default=True, action=BooleanOptionalAction)
    parser.add_argument("--knn_probe", default=False, action='store_true')
    parser = MultiRandMaskCLR.add_model_specific_args(parser)
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

    model = MultiRandMaskCLR(**args.__dict__)

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
